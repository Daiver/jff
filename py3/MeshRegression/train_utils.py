import numpy as np
import torch


def run_validate(model, criterion, valid_loader, device):
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
