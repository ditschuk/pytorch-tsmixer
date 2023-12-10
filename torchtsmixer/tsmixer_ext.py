from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    ConditionalFeatureMixing,
    ConditionalMixerLayer,
    TimeBatchNorm2d,
    feature_to_time,
    time_to_feature,
)


class TSMixerExt(nn.Module):
    """TSMixer model for time series forecasting.

    This model forecasts time series data by integrating historical time series data,
    future known inputs, and static contextual information. It uses a combination of
    conditional feature mixing and mixer layers to process and combine these different
    types of data for effective forecasting.

    Args:
        sequence_length: The length of the input time series sequences.
        prediction_length: The length of the output prediction sequences.
        activation_fn: The name of the activation function to be used.
        num_blocks: The number of mixer blocks in the model.
        dropout_rate: The dropout rate used in the mixer layers.
        input_channels: The number of channels in the historical time series data.
        extra_channels: The number of channels in the extra (future known) inputs.
        hidden_channels: The number of hidden channels used in the mixer layers.
        static_channels: The number of channels in the static feature inputs.
        ff_dim: The inner dimension of the feedforward network in the mixer layers.
        output_channels: The number of output channels for the final output. If None,
                         defaults to the number of input_channels.
        normalize_before: Whether to apply layer normalization before or after mixer layer.
        norm_type: The type of normalization to use. "batch" or "layer".
    """

    def __init__(
        self,
        sequence_length: int,
        prediction_length: int,
        activation_fn: str = "relu",
        num_blocks: int = 2,
        dropout_rate: float = 0.1,
        input_channels: int = 1,
        extra_channels: int = 1,
        hidden_channels: int = 64,
        static_channels: int = 1,
        ff_dim: int = 64,
        output_channels: int = None,
        normalize_before: bool = False,
        norm_type: str = "layer",
    ):
        assert static_channels > 0, "static_channels must be greater than 0"
        super().__init__()

        # Transform activation_fn string to callable function
        if hasattr(F, activation_fn):
            activation_fn = getattr(F, activation_fn)
        else:
            raise ValueError(f"Unknown activation function: {activation_fn}")

        # Transform norm_type to callable
        assert norm_type in {
            "batch",
            "layer",
        }, f"Invalid norm_type: {norm_type}, must be one of batch, layer."
        norm_type = TimeBatchNorm2d if norm_type == "batch" else nn.LayerNorm

        self.fc_hist = nn.Linear(sequence_length, prediction_length)
        self.fc_out = nn.Linear(hidden_channels, output_channels or input_channels)

        self.feature_mixing_hist = ConditionalFeatureMixing(
            sequence_length=prediction_length,
            input_channels=input_channels + extra_channels,
            output_channels=hidden_channels,
            static_channels=static_channels,
            ff_dim=ff_dim,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )
        self.feature_mixing_future = ConditionalFeatureMixing(
            sequence_length=prediction_length,
            input_channels=extra_channels,
            output_channels=hidden_channels,
            static_channels=static_channels,
            ff_dim=ff_dim,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )

        self.conditional_mixer = self._build_mixer(
            num_blocks,
            hidden_channels,
            prediction_length,
            ff_dim=ff_dim,
            static_channels=static_channels,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )

    @staticmethod
    def _build_mixer(
        num_blocks: int, hidden_channels: int, prediction_length: int, **kwargs
    ):
        """Build the mixer blocks for the model."""
        channels = [2 * hidden_channels] + [hidden_channels] * (num_blocks - 1)

        return nn.ModuleList(
            [
                ConditionalMixerLayer(
                    input_channels=in_ch,
                    output_channels=out_ch,
                    sequence_length=prediction_length,
                    **kwargs,
                )
                for in_ch, out_ch in zip(channels[:-1], channels[1:])
            ]
        )

    def forward(
        self,
        x_hist: torch.Tensor,
        x_extra_hist: torch.Tensor,
        x_extra_future: torch.Tensor,
        x_static: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for the TSMixer model.

        Processes historical and future data, along with static features, to produce a
        forecast.

        Args:
            x_hist: Historical time series data (batch_size, sequence_length,
                input_channels).
            x_extra_hist: Additional historical data (batch_size, sequence_length,
                extra_channels).
            x_extra_future: Future known data (batch_size, prediction_length,
                extra_channels).
            x_static: Static contextual data (batch_size, static_channels).

        Returns:
            The output tensor representing the forecast (batch_size, prediction_length,
            output_channels).
        """

        # Concatenate historical time series data with additional historical data
        x_hist = torch.cat([x_hist, x_extra_hist], dim=-1)

        # Transform feature space to time space, apply linear trafo, and convert back
        x_hist_temp = feature_to_time(x_hist)
        x_hist_temp = self.fc_hist(x_hist_temp)
        x_hist = time_to_feature(x_hist_temp)

        # Apply conditional feature mixing to the historical data
        x_hist, _ = self.feature_mixing_hist(x_hist, x_static=x_static)

        # Apply conditional feature mixing to the future data
        x_future, _ = self.feature_mixing_future(x_extra_future, x_static=x_static)

        # Concatenate processed historical and future data
        x = torch.cat([x_hist, x_future], dim=-1)

        # Process the concatenated data through the mixer layers
        for mixing_layer in self.conditional_mixer:
            x = mixing_layer(x, x_static=x_static)

        # Final linear transformation to produce the forecast
        x = self.fc_out(x)

        return x


if __name__ == "__main__":
    sequence_length = 10
    prediction_length = 5

    input_channels = 2
    extra_channels = 3
    hidden_channels = 8
    static_channels = 4
    output_channels = 4

    m = TSMixerExt(
        sequence_length,
        prediction_length,
        input_channels=input_channels,
        extra_channels=extra_channels,
        hidden_channels=hidden_channels,
        static_channels=static_channels,
        output_channels=output_channels,
    )

    x_hist = torch.randn(3, sequence_length, input_channels, requires_grad=True)
    x_extra_hist = torch.randn(3, sequence_length, extra_channels, requires_grad=True)
    x_extra_future = torch.randn(3, prediction_length, extra_channels, requires_grad=True)
    x_static = torch.randn(3, static_channels, requires_grad=True)

    y = m.forward(
        x_hist=x_hist,
        x_extra_hist=x_extra_hist,
        x_extra_future=x_extra_future,
        x_static=x_static,
    )
