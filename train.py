import logging
import math
import os

import tensorflow as tf
from dataclasses import dataclass, field
from transformers import TFTrainingArguments, TFTrainer, LongformerConfig, LongformerTokenizer
from transformers import HfArgumentParser
from algorithm_longformer_pretrainning.algorithms.modeling_tf_longformer import TFLongformerModel
import algorithm_longformer_pretrainning.utils.create_pretraining_data as create_pretraining_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def pretrain_and_evaluate(args, model, eval_only):
    val_dataset = tf.data.TFRecordDataset(file_path=args.val_datapath)
    if eval_only:
        train_dataset = val_dataset
    else:
        logger.info(f'Loading and tokenizing training data is usually slow: {args.train_datapath}')
        train_dataset = tf.data.Dataset(file_path=args.train_datapath)

    trainer = TFTrainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=val_dataset)

    eval_loss = trainer.evaluate()
    eval_loss = eval_loss['eval_loss']
    logger.info(f'Initial eval bpc: {eval_loss / math.log(2)}')

    if not eval_only:
        trainer.train()
        trainer.save_model()

        eval_loss = trainer.evaluate()
        eval_loss = eval_loss['eval_loss']
        logger.info(f'Eval bpc after pretraining: {eval_loss / math.log(2)}')


@dataclass
class ModelArgs:
    attention_window: int = field(default=512, metadata={"help": "Size of attention window"})
    max_pos: int = field(default=4096, metadata={"help": "Maximum position"})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(create_pretraining_data.processors.keys())})
    data_dir: str = field(metadata={"help": "Should contain the data files for the task."})
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def create_pretraining_data(args: DataTrainingArguments):
    need_create_valid_tfrecord_samples = True
    need_create_train_tfrecord_samples = True
    if os.path.exists(args.train_datapath_output):
       need_create_train_tfrecord_samples = False
    if os.path.exists(args.val_datapath_output):
        need_create_valid_tfrecord_samples = False


parser = HfArgumentParser((TFTrainingArguments, ModelArgs, DataTrainingArguments))


training_args, model_args, data_training_args = parser.parse_args_into_dataclasses(look_for_args_file=False, args=[
    '--output_dir', './output',
    '--warmup_steps', '500',
    '--learning_rate', '0.00003',
    '--weight_decay', '0.01',
    '--adam_epsilon', '1e-6',
    '--max_steps', '3000',
    '--logging_steps', '500',
    '--save_steps', '500',
    '--max_grad_norm', '5.0',
    '--per_gpu_eval_batch_size', '8',
    '--per_gpu_train_batch_size', '2',  # 32GB gpu with fp32
    '--gradient_accumulation_steps', '32',
    '--evaluate_during_training',
    '--do_train',
    '--do_eval',

    '--vocab_size', '50000',
    '--hidden_size', '768',
    '--num_hidden_layers', '12',
    '--num_attention_heads', '12',
    '--intermediate_size', '3072',
    '--hidden_act', 'gelu',
    '--hidden_dropout_prob', '0.1',
    '--attention_probs_dropout_prob', '0.1',
    '--max_position_embeddings', '4098',
    '--type_vocab_size', '1',
    '--initializer_range', '0.02',
    '--attention_window', '[512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512]',
    '--bos_token_id', '0',
    '--end_token_id', '2',
    '--pad_token_id', '1',
    '--sep_token_id', '2',
    '--layer_norm_eps', '0.00001',
    '--model_type', 'longformer',
    '--attention_mode', 'longformer'
])
data_training_args.val_datapath = 'wikitext-103/valid.txt'
data_training_args.train_datapath = 'wikitext-103/train.txt'

data_training_args.val_datapath_output = training_args.output_dir + "/wikitext-103/train.tfrecord"
data_training_args.train_datapath = training_args.output_dir + "/wikitext-103/valid.tfrecord"


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    tokenizer = LongformerTokenizer.from_pretrained("longformer-base-4096")
    create_pretraining_data(data_training_args)

    config = LongformerConfig(
            vocab_size=training_args.vocab_size,
            hidden_size=training_args.hidden_size,
            num_hidden_layers=training_args.num_hidden_layers,
            num_attention_heads=training_args.num_attention_heads,
            intermediate_size=training_args.intermediate_size,
            hidden_act=training_args.hidden_act,
            hidden_dropout_prob=training_args.hidden_dropout_prob,
            attention_probs_dropout_prob=training_args.attention_probs_dropout_prob,
            max_position_embeddings=training_args.max_position_embeddings,
            type_vocab_size=training_args.type_vocab_size,
            initializer_range=training_args.initializer_range,
            attention_window=training_args.attention_window,
        )

    # long_formeformer_tokernizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = TFLongformerModel(confg=config)
    logger.info('Evaluating roberta-base (seqlen: 4096) for refernece ...')
    pretrain_and_evaluate(training_args, model, eval_only=False)

