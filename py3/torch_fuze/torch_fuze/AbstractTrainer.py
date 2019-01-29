import abc
import torch


class AbstractTrainer:
    def __init__(self):
        pass

    @abc.abstractmethod()
    def run(self, *args, **kwargs):
        pass
