import abc
import torch


class AbstractEvaluator:
    def __init__(self):
        pass

    @abc.abstractmethod()
    def run(self, *args, **kwargs):
        pass


"""
with torch.no_grad():
        model = model.to(device)
        model.eval()
        losses = []
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
    return np.mean(losses)
"""
