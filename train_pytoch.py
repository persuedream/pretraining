import logging
import math
from dataclasses import dataclass, field
from transformers import RobertaForMaskedLM, RobertaTokenizerFast, TextDataset, DataCollatorForLanguageModeling, \
    Trainer, BertTokenizer, BertTokenizerFast
from transformers import TrainingArguments, HfArgumentParser
from transformers.modeling_longformer import LongformerSelfAttention

import os

from algorithm_longformer_pretrainning.algorithms.custom_trainer import CustomTrainer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from algorithm_longformer_pretrainning.algorithms.data_set import LineByLineTextDataset, CustomIterableDataset

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S', level=logging.INFO)


class RobertaLongSelfAttention(LongformerSelfAttention):
    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
    ):
        return super().forward(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)


class RobertaLongForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.roberta.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = RobertaLongSelfAttention(config, layer_id=i)


def create_long_model(save_model_to, attention_window, max_pos):
    model = RobertaForMaskedLM.from_pretrained('./pretrained_model/roberta_chinese_base')
    tokenizer = BertTokenizerFast.from_pretrained('./pretrained_model/roberta_chinese_base', model_max_length=max_pos)
    config = model.config

    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.roberta.embeddings.position_embeddings.weight.shape
    max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
    config.max_position_embeddings = max_pos
    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.roberta.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_pos - 2
    while k < max_pos - 1 - step:
        new_pos_embed[k:(k + step)] = model.roberta.embeddings.position_embeddings.weight[2:]
        k += step
    model.roberta.embeddings.position_embeddings.weight.data = new_pos_embed

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.roberta.encoder.layer):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = layer.attention.self.query
        longformer_self_attn.key = layer.attention.self.key
        longformer_self_attn.value = layer.attention.self.value

        longformer_self_attn.query_global = layer.attention.self.query
        longformer_self_attn.key_global = layer.attention.self.key
        longformer_self_attn.value_global = layer.attention.self.value

        layer.attention.self = longformer_self_attn

    logger.info(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    return model, tokenizer


def copy_proj_layers(model):
    for i, layer in enumerate(model.roberta.encoder.layer):
        layer.attention.self.query_global = layer.attention.self.query
        layer.attention.self.key_global = layer.attention.self.key
        layer.attention.self.value_global = layer.attention.self.value
    return model


def pretrain_and_evaluate(args, model, tokenizer, eval_only, model_path):

    val_dataset, train_dataset = None, None
    for val_datapath in args.val_datapath:
        if not val_dataset:
            val_dataset = CustomIterableDataset(tokenizer=tokenizer,
                                                file_path=val_datapath,
                                                block_size=model.config.max_position_embeddings)
        else:
            val_dataset += CustomIterableDataset(tokenizer=tokenizer, file_path=val_datapath,
                                                 block_size=model.config.max_position_embeddings)
    if eval_only:
        train_dataset = val_dataset
    else:
        logger.info(f'Loading and tokenizing training data is usually slow: {args.train_datapath}')
        for train_datapath in args.train_datapath:
            if not train_dataset:
                train_dataset = CustomIterableDataset(tokenizer=tokenizer,
                                                      file_path=train_datapath,
                                                      block_size=model.config.max_position_embeddings,)
            else:
                train_dataset += CustomIterableDataset(tokenizer=tokenizer,
                                                      file_path=train_datapath,
                                                      block_size=model.config.max_position_embeddings,)
        # train_dataset = val_dataset

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    trainer = CustomTrainer(model=model, args=args, data_collator=data_collator,
                      train_dataset=train_dataset, eval_dataset=val_dataset, prediction_loss_only=True, )

    eval_loss = trainer.evaluate()
    eval_loss = eval_loss['eval_loss']
    logger.info(f'Initial eval bpc: {eval_loss / math.log(2)}')

    if not eval_only:
        trainer.train(model_path=model_path)
        trainer.save_model()

        eval_loss = trainer.evaluate()
        eval_loss = eval_loss['eval_loss']
        logger.info(f'Eval bpc after pretraining: {eval_loss / math.log(2)}')


@dataclass
class ModelArgs:
    attention_window: int = field(default=32, metadata={"help": "Size of attention window"})
    max_pos: int = field(default=512, metadata={"help": "Maximum position"})


parser = HfArgumentParser((TrainingArguments, ModelArgs,))
training_args, model_args = parser.parse_args_into_dataclasses(look_for_args_file=False, args=[
    '--output_dir', './output',
    '--warmup_steps', '500',
    '--learning_rate', '0.00003',
    '--weight_decay', '0.01',
    '--adam_epsilon', '1e-6',
    '--max_steps', '3000',
    '--logging_steps', '500',
    '--save_steps', '500',
    '--max_grad_norm', '5.0',
    '--per_gpu_eval_batch_size', '2',
    '--per_gpu_train_batch_size', '1',  # 32GB gpu with fp32
    '--gradient_accumulation_steps', '1',
    '--evaluate_during_training',
    '--do_train',
    '--do_eval',
])

training_args.val_datapath = ['/media/txguo/866225e9-e15c-2d46-aba5-1ce1c0452e49/download_pdf/another_samples/1_5001.txt',
                              '/media/txguo/866225e9-e15c-2d46-aba5-1ce1c0452e49/download_pdf/another_samples/5001_10001.txt']
training_args.train_datapath = ['/media/txguo/866225e9-e15c-2d46-aba5-1ce1c0452e49/download_pdf/another_samples/10001_15001.txt',
                                '/media/txguo/866225e9-e15c-2d46-aba5-1ce1c0452e49/download_pdf/another_samples/15001_20001.txt']

if __name__ == "__main__":
    roberta_base = RobertaForMaskedLM.from_pretrained('./pretrained_model/roberta_chinese_base')
    roberta_base_tokenizer = BertTokenizerFast.from_pretrained('./pretrained_model/roberta_chinese_base')
    # logger.info('Evaluating roberta-base (seqlen: 512) for refernece ...')
    # pretrain_and_evaluate(training_args, roberta_base, roberta_base_tokenizer, eval_only=True, model_path=None)

    model_path = f'{training_args.output_dir}/roberta-base-{model_args.max_pos}'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    logger.info(f'Converting roberta-base into roberta-base-{model_args.max_pos}')
    model, tokenizer = create_long_model(
        save_model_to=model_path, attention_window=model_args.attention_window, max_pos=model_args.max_pos)

    logger.info(f'Loading the model from {model_path}')
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = RobertaLongForMaskedLM.from_pretrained(model_path)

    logger.info(f'Pretraining roberta-base-{model_args.max_pos} ... ')

    training_args.max_steps = 4  ## <<<<<<<<<<<<<<<<<<<<<<<< REMOVE THIS <<<<<<<<<<<<<<<<<<<<<<<<

    tokenizer._add_tokens(["[unused1]"], special_tokens=True)
    pretrain_and_evaluate(training_args, model, tokenizer, eval_only=False, model_path=training_args.output_dir)

    logger.info(f'Copying local projection layers into global projection layers ... ')
    model = copy_proj_layers(model)
    logger.info(f'Saving model to {model_path}')
    model.save_pretrained(model_path)
