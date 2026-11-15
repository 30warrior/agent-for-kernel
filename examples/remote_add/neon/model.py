"""Handwritten wrapper for the remote Add smoke example."""
import torch


class ModelNew(torch.nn.Module):
    def __init__(self, reference, extension):
        super().__init__()
        self.extension = extension

    def forward(self, input1, input2):
        return self.extension.add(input1, input2)
