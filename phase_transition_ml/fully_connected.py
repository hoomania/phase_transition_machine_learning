import torch
import torch.nn.functional as nnf
import torch.nn as nn


class FCN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(64, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, result):
        result = nnf.relu(self.fc1(result))
        result = torch.sigmoid(self.fc2(result))
        return result


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# model = FCN()
# print(model)
