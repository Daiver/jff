from .abstract_trainer import AbstractTrainer


class SupervisedTrainer(AbstractTrainer):
    def __init__(self):
        super().__init__()

    def run(self, *args, **kwargs):
        pass
