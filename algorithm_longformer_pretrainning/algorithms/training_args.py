import logging
from functools import cached_property
from typing import Tuple

from transformers import TrainingArguments
from transformers.file_utils import tf_required, is_tf_available

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


class TFTrainingArguments(TrainingArguments):
    """
    实现对tensorflow的参数配置
    """
    @cached_property
    @tf_required
    def _setup_devices(self) -> Tuple["tf.device", int]:
        logger.info("tensorflow: setting up devices")
        if self.no_cuda:
            device = tf.device("cpu")
            n_gpu = 0
        elif tf.config.list_physical_devices("tpu"):
            device = tf.device("tpu")
            n_gpu = 0
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = tf.device("cuda:0" if tf.config.list_physical_devices("gpu") else "cpu")
            n_gpu = len(tf.config.list_physical_devices("GPU"))
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            device = tf.device("cuda", self.local_rank)
            n_gpu = 1

        return device, n_gpu

    @property
    @tf_required
    def device(self) -> "tf.device":
        """
        The device used by this process.
        """
        return self._setup_devices[0]

    @property
    @tf_required
    def n_gpu(self):
        """
        The number of GPUs used by this process.

        Note:
            This will only be greater than one when you have multiple GPUs available but are not using distributed
            training. For distributed training, it will always be 1.
        """
        return self._setup_devices[1]
