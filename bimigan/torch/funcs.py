import logging
import random
import torch
import numpy as np

from torch import nn

LOGGER = logging.getLogger('bimigan.torch')

def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_torch_device_smart():
    if torch.cuda.is_available():
        print("device is cuda")
        return 'cuda'
    print("device is cpu")
    return 'cpu'

def prepare_model(model, device):
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        LOGGER.warning(
            "Multiple (%d) GPUs found. Using Data Parallelism",
            torch.cuda.device_count()
        )
        model = nn.DataParallel(model)

    return model

