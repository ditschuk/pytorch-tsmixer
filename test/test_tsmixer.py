from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn
from hypothesis import given
from hypothesis import strategies as st

from torchtsmixer import TSMixer  # Adjust the import as necessary


class TSMixerTest(unittest.TestCase):
    @given(
        sequence_length=st.integers(min_value=10, max_value=100),
        prediction_length=st.integers(min_value=5, max_value=50),
        input_channels=st.integers(min_value=1, max_value=10),
        num_blocks=st.integers(min_value=1, max_value=5),
        batch_size=st.integers(min_value=1, max_value=10),
    )
    def test_tsmixer_shapes(
        self,
        sequence_length: int,
        prediction_length: int,
        input_channels: int,
        num_blocks: int,
        batch_size: int,
    ) -> None:
        """Test the output shape of TSMixer model.

        Ensures the output shape of the model matches the expected dimensions.

        Args:
            sequence_length: Length of the input time series.
            prediction_length: Length of the output predictions.
            input_channels: Number of channels in the input.
            num_blocks: Number of blocks in the model.
            batch_size: Size of each input batch.
        """
        model = TSMixer(
            sequence_length=sequence_length,
            prediction_length=prediction_length,
            input_channels=input_channels,
            num_blocks=num_blocks,
        )

        x = torch.randn(batch_size, sequence_length, input_channels)
        output = model(x)

        self.assertEqual(output.shape, (batch_size, prediction_length, input_channels))

    def test_tsmixer_sinusoidal_fit(self) -> None:
        """Test if TSMixer can overfit a sinusoidal function.

        This test checks if the model can learn a simple sinusoidal pattern.
        """
        sequence_length = 50
        prediction_length = 10
        input_channels = 1
        num_samples = 100
        epochs = 200

        model = TSMixer(
            sequence_length=sequence_length,
            prediction_length=prediction_length,
            input_channels=input_channels,
        )

        # Generate sinusoidal data
        t = np.linspace(0, 2 * np.pi, sequence_length + prediction_length)
        x = np.sin(t).reshape((1, -1, 1)).repeat(num_samples, axis=0)
        x_tensor, y_tensor = map(
            torch.Tensor, (x[:, :-prediction_length, :], x[:, -prediction_length:, :])
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        final_loss = None
        for _ in range(epochs):
            optimizer.zero_grad()
            output = model(x_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        self.assertLess(final_loss, 0.1)


if __name__ == "__main__":
    unittest.main()
