from typing import List
import itertools
import logging
import os

logger = logging.getLogger("System-GPU")


def gpu_count():
    import torch
    return torch.cuda.device_count()

def get_gpu_device() -> List[str]:
    """
    Returns:
        List of assigned devices.
    """

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        return ["cpu"]
    elif os.environ["CUDA_VISIBLE_DEVICES"] == "":
        return ["cpu"]
    else:
        num_gpu_used = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        assert num_gpu_used > 0, "Invalid CUDA_VISIBLE_DEVICES" 
        return [f"cuda:{i}" for i in range(num_gpu_used)]


def set_cuda_device(device):
    """Set the default cuda-device. Useful on multi-gpu nodes. Should be called in every gpu-thread.
    """
    logger.info(f"Setting device to {device}.")
    if device != "cpu":
        import torch
        torch.cuda.set_device(device)

