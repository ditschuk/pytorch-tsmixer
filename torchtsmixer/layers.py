from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TimeBatchNorm1d(nn.BatchNorm1d):
    """A batch normalization applied to the time dimension of a sequence."""

    def forward(self, x: Tensor) -> Tensor:
        """

        Args:
            x: A 3D tensor with shape (N, S, L) where S is the sequence dimension,
               and L is the feature dimension length.

        Returns:
            A 3D tensor, where the batch normalization has been applied to the
            channel dimension.

        Raises:
            AssertionError: If the input tensor is not 3D.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected 3D x tensor, but got {x.ndim}D tensor instead.")

        x = x.permute(0, 2, 1)
        x = super().forward(x)
        return x.permute(0, 2, 1)


class FeatureMixing(nn.Module):
    """A module for feature mixing with flexibility in normalization and activation.

    This module provides options for batch normalization before or after mixing features,
    uses dropout for regularization, and allows for different activation functions.

    Args:
        input_channels: The number of input channels to the module.
        output_channels: The number of output channels from the module.
        ff_dim: The dimension of the feed-forward network internal to the module.
        activation_fn: The activation function used within the feed-forward network.
        dropout_rate: The dropout probability used for regularization.
        normalize_before: A boolean indicating whether to apply normalization before
            the rest of the operations.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        ff_dim: int,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
        norm_type: type[nn.Module] = TimeBatchNorm1d,
    ):
        """Initializes the FeatureMixing module with the provided parameters."""
        super().__init__()

        self.norm_before = norm_type(input_channels) if normalize_before else nn.Identity()
        self.norm_after = (
            norm_type(output_channels) if not normalize_before else nn.Identity()
        )

        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(input_channels, ff_dim)
        self.fc2 = nn.Linear(ff_dim, output_channels)

        self.projection = (
            nn.Linear(input_channels, output_channels)
            if input_channels != output_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the FeatureMixing module.

        Args:
            x: A 3D tensor with shape (N, C, L) where C is the channel dimension.

        Returns:
            The output tensor after feature mixing.
        """
        x_proj = self.projection(x)

        x = self.norm_before(x)

        x = self.fc1(x)  # Apply the first linear transformation.
        x = self.activation_fn(x)  # Apply the activation function.
        x = self.dropout(x)  # Apply dropout for regularization.
        x = self.fc2(x)  # Apply the second linear transformation.
        x = self.dropout(x)  # Apply dropout again if needed.

        x = x_proj + x  # Add the projection shortcut to the transformed features.

        return self.norm_after(x)


class ConditionalFeatureMixing(nn.Module):
    """Conditional feature mixing module that incorporates static features.

    This module extends the feature mixing process by including static features. It uses
    a linear transformation to integrate static features into the dynamic feature space,
    then applies the feature mixing on the concatenated features.

    Args:
        input_channels: The number of input channels of the dynamic features.
        output_channels: The number of output channels after feature mixing.
        static_channels: The number of channels in the static feature input.
        ff_dim: The inner dimension of the feedforward network used in feature mixing.
        activation_fn: The activation function used in feature mixing.
        dropout_rate: The dropout probability used in the feature mixing operation.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        static_channels: int,
        ff_dim: int,
        activation_fn: Callable = F.relu,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.fr_static = nn.Linear(static_channels, output_channels)
        self.fm = FeatureMixing(
            input_channels + output_channels,
            output_channels,
            ff_dim,
            activation_fn,
            dropout_rate,
            normalize_before=False,
            norm_type=nn.LayerNorm,
        )

    def forward(
        self, x: torch.Tensor, x_static: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies conditional feature mixing using both dynamic and static inputs.

        Args:
            x: A tensor representing dynamic features, typically with shape
               [batch_size, time_steps, input_channels].
            x_static: A tensor representing static features, typically with shape
               [batch_size, static_channels].

        Returns:
            A tuple containing:
            - The output tensor after applying conditional feature mixing.
            - The transformed static features tensor for monitoring or further processing.
        """
        v = self.fr_static(x_static)  # Transform static features to match output channels.
        v = v.unsqueeze(1).repeat(
            1, x.shape[1], 1
        )  # Repeat static features across time steps.

        return (
            self.fm(
                torch.cat([x, v], dim=-1)
            ),  # Apply feature mixing on concatenated features.
            v.detach(),  # Return detached static feature for monitoring or further use.
        )


class TimeMixing(nn.Module):
    """Applies a transformation over the time dimension of a sequence.

    This module applies a linear transformation followed by an activation function
    and dropout over the sequence length of the input feature tensor after converting
    feature maps to the time dimension and then back.

    Args:
        input_channels: The number of input channels to the module.
        sequence_length: The length of the sequences to be transformed.
        activation_fn: The activation function to be used after the linear transformation.
        dropout_rate: The dropout probability to be used after the activation function.
    """

    def __init__(
        self,
        input_channels: int,
        sequence_length: int,
        activation_fn: Callable = F.relu,
        dropout_rate: float = 0.1,
        norm_type: type[nn.Module] = TimeBatchNorm1d,
    ):
        """Initializes the TimeMixing module with the specified parameters."""
        super().__init__()
        self.norm = norm_type(input_channels)  # Assuming a dummy channel dimension
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(sequence_length, sequence_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the time mixing operations on the input tensor.

        Args:
            x: A 3D tensor with shape (N, C, L), where C = channel dimension and
                L = sequence length.

        Returns:
            The normalized output tensor after time mixing transformations.
        """
        x_temp = feature_to_time(
            x
        )  # Convert feature maps to time dimension. Assumes definition elsewhere.
        x_temp = self.activation_fn(self.fc1(x_temp))
        x_temp = self.dropout(x_temp)
        x_res = time_to_feature(x_temp)  # Convert back from time to feature maps.

        return self.norm(x + x_res)  # Apply normalization and combine with original input.


class MixerLayer(nn.Module):
    """A residual block that combines time and feature mixing for sequence data.

    This module sequentially applies time mixing and feature mixing, which are forms
    of data augmentation and feature transformation that can help in learning temporal
    dependencies and feature interactions respectively.

    Args:
        input_channels: The number of input channels to the module.
        sequence_length: The length of the input sequences.
        ff_dim: The inner dimension of the feedforward network used in feature mixing.
        activation_fn: The activation function used in both time and feature mixing.
        dropout_rate: The dropout probability used in both mixing operations.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        sequence_length: int,
        ff_dim: int,
        activation_fn: Callable = F.relu,
        dropout_rate: float = 0.1,
    ):
        """Initializes the MixLayer with time and feature mixing modules."""
        super().__init__()

        self.time_mixing = TimeMixing(
            input_channels, sequence_length, activation_fn, dropout_rate
        )
        self.feature_mixing = FeatureMixing(
            input_channels, output_channels, ff_dim, activation_fn, dropout_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MixLayer module.

        Args:
            x: A 3D tensor with shape (N, C, L) to be processed by the mixing layers.

        Returns:
            The output tensor after applying time and feature mixing operations.
        """
        x = self.time_mixing(x)  # Apply time mixing first.
        x = self.feature_mixing(x)  # Then apply feature mixing.

        return x


class ConditionalMixerLayer(nn.Module):
    """Conditional mix layer combining time and feature mixing with static context.

    This module combines time mixing and conditional feature mixing, where the latter
    is influenced by static features. This allows the module to learn representations
    that are influenced by both dynamic and static features.

    Args:
        input_channels: The number of input channels of the dynamic features.
        output_channels: The number of output channels after feature mixing.
        static_channels: The number of channels in the static feature input.
        sequence_length: The length of the input sequences.
        ff_dim: The inner dimension of the feedforward network used in feature mixing.
        activation_fn: The activation function used in both mixing operations.
        dropout_rate: The dropout probability used in both mixing operations.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        static_channels: int,
        sequence_length: int,
        ff_dim: int,
        activation_fn: Callable = F.relu,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.time_mixing = TimeMixing(
            input_channels,
            sequence_length,
            activation_fn,
            dropout_rate,
            norm_type=nn.LayerNorm,
        )
        self.feature_mixing = ConditionalFeatureMixing(
            input_channels,
            output_channels=output_channels,
            static_channels=static_channels,
            ff_dim=ff_dim,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        """Forward pass for the conditional mix layer.

        Args:
            x: A tensor representing dynamic features, typically with shape
               [batch_size, time_steps, input_channels].
            x_static: A tensor representing static features, typically with shape
               [batch_size, static_channels].

        Returns:
            The output tensor after applying time and conditional feature mixing.
        """
        x = self.time_mixing(x)  # Apply time mixing first.
        x, _ = self.feature_mixing(x, x_static)  # Then apply conditional feature mixing.

        return x


def time_to_feature(x: torch.Tensor) -> torch.Tensor:
    """Converts a time series tensor to a feature tensor."""
    return x.permute(0, 2, 1)


feature_to_time = time_to_feature
