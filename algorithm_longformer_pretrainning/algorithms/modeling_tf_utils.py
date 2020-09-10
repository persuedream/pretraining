import warnings

import tensorflow as tf
from transformers import shape_list


class TFCausalLanguageModelingLoss:
    """
    Loss function suitable for causal language modeling (CLM), that is, the task of guessing the next token.

    .. note::

        Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.

    """

    def compute_loss(self, labels, logits):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        # make sure only labels that are not equal to -100
        # are taken into account as loss
        active_loss = tf.reshape(labels, (-1,)) != -100
        reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
        labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)
        return loss_fn(labels, reduced_logits)


class TFMaskedLanguageModelingLoss(TFCausalLanguageModelingLoss):
    """
   Loss function suitable for masked language modeling (MLM), that is, the task of guessing the masked tokens.

   .. note::

        Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.

"""
