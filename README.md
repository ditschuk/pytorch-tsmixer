# TSMixer: Time Series Mixer for Forecasting

## Overview
TSMixer is an **unofficial** PyTorch-based implementation of the TSMixer architecture as described [TSMixer Paper](https://arxiv.org/pdf/2303.06053.pdf). It leverages mixer layers for processing time series data, offering a robust approach for both standard and extended forecasting tasks.

## Installation
To install the necessary dependencies, run:
```bash
pip install -e .
```

## Modules
- `tsmixer.py`: Contains the `TSMixer` class, a model using mixer layers for time series forecasting.
- `tsmixer_ext.py`: Implements the `TSMixerExt` class, an extended version of TSMixer that integrates additional inputs and contextual information.

## Usage

### TSMixer
```python
from torchtsmixer import TSMixer
import torch

m = TSMixer(sequence_length=10, prediction_length=5, input_channels=2, output_channels=4)
x = torch.randn(3, 10, 2)
y = m(x)
```

### TSMixerExt
```python
from torchtsmixer import TSMixerExt
import torch

m = TSMixerExt(
    sequence_length=10,
    prediction_length=5,
    input_channels=2,
    extra_channels=3,
    hidden_channels=8,
    static_channels=4,
    output_channels=4
)

x_hist = torch.randn(3, 10, 2, requires_grad=True)
x_extra_hist = torch.randn(3, 10, 3, requires_grad=True)
x_extra_future = torch.randn(3, 5, 3, requires_grad=True)
x_static = torch.randn(3, 4, requires_grad=True)

y = m.forward(
    x_hist=x_hist,
    x_extra_hist=x_extra_hist,
    x_extra_future=x_extra_future,
    x_static=x_static
)
```

## Example: Training Loop with TSMixer

Here's a basic example of how to use `TSMixer` in a simple training loop. This example assumes a regression task with a mean squared error loss and an Adam optimizer.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchtsmixer import TSMixer

# Model parameters
sequence_length = 10
prediction_length = 5
input_channels = 2
output_channels = 1

# Create the TSMixer model
model = TSMixer(sequence_length, prediction_length, input_channels, output_channels)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Dummy dataset (replace with real data)
# Assuming batch_size, seq_len, num_features format
X_train = torch.randn(10,32, sequence_length, input_channels)
y_train = torch.randn(10,32, prediction_length, output_channels)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for X,y in zip(X_train, y_train):
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete")
```

This example is quite basic and should be adapted to your specific dataset and task. For instance, you might want to add data loading with `DataLoader`, validation steps, and more sophisticated training logic.

## Testing
Run tests using:
```bash
python -m unittest
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
This implementation is based on the TSMixer model as described in [TSMixer Paper](https://arxiv.org/pdf/2303.06053.pdf).
