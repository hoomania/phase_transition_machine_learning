import torch
import torch.nn as nn
import torch.nn.functional as nnf

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[0], [1], [0], [1]], dtype=torch.float32)

n_samples, n_features = X.shape
print(f'#samples: {n_samples}, #features: {n_features}')

X_test = torch.tensor([5], dtype=torch.float32)

input_size = n_features
output_size = n_features


class FCN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 8)
        self.fc2 = nn.Linear(8, output_dim)

    def forward(self, x):
        # x = torch.relu(self.fc1(x))
        x = nnf.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


model = FCN(input_size, output_size)

print(f'prediction before training: f(5) = {model(X_test).item():.3f}')

learning_rate = 0.01
iterations = 500

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(iterations):
    # forward
    y_pred = model(X)

    # loss
    lss = loss(Y, y_pred)

    # gradient
    lss.backward()

    # update weights
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 100 == 0:
        print(model(X_test))
        # [w, b] = model.parameters()
        # print(f'epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss = {lss:.8f}')

print(f'prediction after training: f(5) = {model(X_test).item():.3f}')
