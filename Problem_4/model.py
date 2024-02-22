import torch
import torch.nn as nn


def SimpleNNRelu():
    return nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
    )


def SimpleNNHardTanh():
    return nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Hardtanh(),
            nn.Linear(128, 64),
            nn.Hardtanh(),
            nn.Linear(64, 10)
    )
