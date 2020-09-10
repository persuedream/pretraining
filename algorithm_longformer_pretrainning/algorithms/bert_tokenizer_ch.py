"""
针对中文的bert分词模型.
主要功能:
1. 将分词信息添加的与训练模型中,一个完整的词之间的字通过"##"连接,eg:
    中国 --> 中##国
    北京 --> 北##京
"""
import re
import logging

import jieba
from transformers import BertTokenizer, WordpieceTokenizer
from transformers.tokenization_bert import whitespace_tokenize

CH_RE = re.compile("[\u4E00-\u9FA5]", re.I)
logger = logging.getLogger(__file__)


class CHBertTokenizer(BertTokenizer):
    """
    针对中文的分词
    """

    def __init__(
            self,
            vocab_file,
            do_lower_case=True,
            do_basic_tokenize=True,
            never_split=None,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            tokenize_chinese_chars=False,
            **kwargs
    ):
        super().__init__(vocab_file,
                         do_lower_case=do_lower_case,
                         do_basic_tokenize=do_basic_tokenize,
                         never_split=never_split,
                         unk_token=unk_token,
                         sep_token=sep_token,
                         pad_token=pad_token,
                         cls_token=cls_token,
                         mask_token=mask_token,
                         tokenize_chinese_chars=tokenize_chinese_chars,
                         **kwargs)
        self.wordpiece_tokenizer = CHTokenzier(vocab=self.vocab, unk_token=self.unk_token)

    def _tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            tokens = self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens)
            for token in tokens:

                # If the token is part of the never_split set
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens


class CHTokenzier(WordpieceTokenizer):
    """
    针对中文的分词模块
    """
    count = 0

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """
        seq_cws_dict = {}
        output_tokens = []
        if self.count % 10000 == 0:
            logger.info(f"count ={self.count}, processing text: {text}")
        self.count += 1
        for ind, token in enumerate(whitespace_tokenize(text)):
            seq_cws = jieba.lcut(token)
            seq_cws_dict.update({x: 1 for x in seq_cws})

        for token in whitespace_tokenize(text):

            chars = list(token)
            i = 0

            while i < len(chars):
                if len(CH_RE.findall(chars[i])) == 0:  # 不是中文的，原文加进去。
                    output_tokens.append(token)
                    break

                has_add = False
                for length in range(5, 0, -1):
                    if i + length > len(chars):
                        continue
                    if ''.join(chars[i:i + length]) in seq_cws_dict:
                        output_tokens.append(chars[i])
                        for l in range(1, length):
                            output_tokens.append('##' + chars[i + l])
                        i += length
                        has_add = True
                        break
                if not has_add:
                    output_tokens.append(chars[i])
                    i += 1

        return output_tokens
