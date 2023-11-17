import unittest

import torch
from hypothesis import given
from hypothesis import strategies as st

from torchtsmixer import TSMixerExt  # Replace with the actual import


class TestTSMixerExt(unittest.TestCase):
    @given(
        batch_size=st.integers(min_value=1, max_value=10),
        sequence_length=st.integers(min_value=1, max_value=20),
        prediction_length=st.integers(min_value=1, max_value=20),
        input_channels=st.integers(min_value=1, max_value=5),
        extra_channels=st.integers(min_value=1, max_value=5),
        static_channels=st.integers(min_value=1, max_value=5),
        hidden_channels=st.integers(min_value=1, max_value=64),
        output_channels=st.integers(min_value=1, max_value=5),
    )
    def test_output_shape(
        self,
        batch_size: int,
        sequence_length: int,
        prediction_length: int,
        input_channels: int,
        extra_channels: int,
        static_channels: int,
        hidden_channels: int,
        output_channels: int,
    ) -> None:
        """Test the output shape of TSMixerExt model.

        Args:
            batch_size: Batch size for input tensors.
            sequence_length: Length of the input time series.
            prediction_length: Length of the output predictions.
            input_channels: Number of input channels.
            extra_channels: Number of extra input channels.
            static_channels: Number of static input channels.
            hidden_channels: Number of hidden channels in the model.
            output_channels: Number of output channels.
        """
        model = TSMixerExt(
            sequence_length=sequence_length,
            prediction_length=prediction_length,
            input_channels=input_channels,
            extra_channels=extra_channels,
            hidden_channels=hidden_channels,
            static_channels=static_channels,
            output_channels=output_channels,
        )

        x_hist = torch.randn(batch_size, sequence_length, input_channels)
        x_extra_hist = torch.randn(batch_size, sequence_length, extra_channels)
        x_extra_future = torch.randn(batch_size, prediction_length, extra_channels)
        x_static = torch.randn(batch_size, static_channels)

        output = model(x_hist, x_extra_hist, x_extra_future, x_static)
        self.assertEqual(output.shape, (batch_size, prediction_length, output_channels))

    def test_overfitting_on_sinusoidal_function(self) -> None:
        """Test the model's ability to overfit a sinusoidal function."""
        torch.seed()
        sequence_length = 10
        prediction_length = 5
        batch_size = 32
        epochs = 300
        learning_rate = 0.01
        static_channels = 4
        input_channels = 2

        model = TSMixerExt(
            sequence_length=sequence_length,
            prediction_length=prediction_length,
            static_channels=static_channels,
            extra_channels=input_channels,
            input_channels=input_channels,
        )
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        t = torch.linspace(0, 20, steps=sequence_length + prediction_length)
        y_true = torch.sin(t).unsqueeze(1).repeat(batch_size, 1, input_channels)
        x_hist = y_true[:, :sequence_length, :]
        x_target = y_true[:, sequence_length:, :]
        x_extra_hist = torch.zeros_like(x_hist)
        x_extra_future = torch.zeros(batch_size, prediction_length, y_true.size(2))
        x_static = torch.randn(batch_size, static_channels)

        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            predictions = model(x_hist, x_extra_hist, x_extra_future, x_static)
            loss = criterion(predictions, x_target)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            predictions = model(x_hist, x_extra_hist, x_extra_future, x_static)
            self.assertTrue(torch.allclose(predictions, x_target, atol=0.1))


if __name__ == "__main__":
    unittest.main()
