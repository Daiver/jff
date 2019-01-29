from .abstract_trainer import AbstractTrainer


class SupervisedTrainer(AbstractTrainer):
    def __init__(self, model, criterion, device="cuda"):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.device = device

    def run(self,
            loader,
            optimizer,
            start_epoch,
            end_epoch,
            callbacks=None):
        callbacks = [] if callbacks is None else callbacks

