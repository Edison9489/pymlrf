from dataclasses import dataclass
import torch
from typing import Dict

__all__ = [
    "DatasetOutput"
    ]

@dataclass
class DatasetOutput:
    """
    Class for defining standardized output from DataLoaders as a tuple format,
    where the first element of the tuple is the input data to the model, and the
    second element is the target data for the loss function. This format is commonly
    used for straightforward model training pipelines in PyTorch.

    Attributes:
        inputs (torch.Tensor): The input tensors to the model.
        targets (torch.Tensor): The target tensors for the loss calculation.
    """
    inputs: torch.Tensor
    targets: torch.Tensor
