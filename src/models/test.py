import torch
import torch.nn as nn

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'models'))
from amr_transformer import amrTransformer
torch.manual_seed(42)

B = 16
D = 128
K = 5

# Four placeholder observation embeddings
observations = torch.zeros(B, 4, D)

# Random goals in robot-relative coordinates
goal_xy = torch.empty(B, 2).uniform_(-2.0, 2.0)
goal_xy[:, 0] += 3.0  # Keep goals generally ahead of robot

# Encode goal coordinates in the first two feature dimensions
goals = torch.zeros(B, 1, D)
goals[:, 0, :2] = goal_xy

# Five evenly spaced future waypoints
fractions = torch.linspace(1 / K, 1.0, K)

targets = fractions[None, :, None] * goal_xy[:, None, :]

print("Observations:", observations.shape)
print("Goals:", goals.shape)
print("Targets:", targets.shape)
print("First goal:", goal_xy[0])
print("First trajectory:", targets[0])

model = amrTransformer(embed_dim=128, num_waypoints=5)

# Disable dropout for this deterministic overfitting test
model.eval()

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.0
)

for step in range(501):

    # TODO 1: Reset gradients
    optimizer.zero_grad()

    # TODO 2: Forward pass
    predictions = model(observations, goals)

    # TODO 3: Compute MSE between predictions and targets
    loss = criterion(predictions, targets)

    # TODO 4: Backpropagate and update parameters
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step:3d} | Loss: {loss.item():.6f}")

with torch.no_grad():
    predictions = model(observations, goals)

print("Goal A:", goal_xy[0])
print("Predicted A:\n", predictions[0])

print("Goal B:", goal_xy[1])
print("Predicted B:\n", predictions[1])