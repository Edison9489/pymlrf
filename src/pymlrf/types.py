from typing import Protocol, List, Literal, Any, Tuple
import torch
from torch.utils.data import DataLoader
import logging

from .SupervisedLearning.torch.Dataset import DatasetOutput


class TrainSingleEpochProtocol(Protocol):
        
    def __call__(
        self, 
        model:torch.nn.Module, 
        data_loader:DataLoader, 
        gpu:bool, 
        optimizer:torch.optim.Optimizer,
        criterion:torch.nn.modules.loss, 
        logger:logging.Logger
        ) -> List[torch.Tensor]:
        ...


class ValidateSingleEpochProtocol(Protocol):
    
    def __call__(
        self, 
        model:torch.nn.Module, 
        data_loader:DataLoader,
        gpu:Literal[True, False], 
        criterion:torch.nn.Module
        ) -> List[torch.Tensor]:
        ...


class GenericDataLoaderProtocol(Protocol):
    
    def __iter__(self)->"GenericDataLoaderProtocol":
        ...

    def __next__(self)->DatasetOutput:
        ...


class CriterionProtocol(Protocol):

    def __call__(self, *args: Any, **kwds: Any) -> torch.Tensor:
        ...