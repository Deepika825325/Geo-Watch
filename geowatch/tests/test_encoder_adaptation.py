"""Tests for GeoWatch multispectral ResNet input adaptation.

The deterministic tests verify ImageNet RGB-to-Sentinel band mapping
without downloading model weights or accessing OSCD data.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.models.encoder import (
    EncoderError,
    adapt_first_convolution,
    resolve_band_names,
)


@pytest.mark.parametrize(
    (
        "input_channels",
        "expected_bands",
    ),
    (
        (
            4,
            (
                "B02",
                "B03",
                "B04",
                "B08",
            ),
        ),
        (
            6,
            (
                "B02",
                "B03",
                "B04",
                "B08",
                "B11",
                "B12",
            ),
        ),
    ),
)
def test_default_band_resolution(
    input_channels: int,
    expected_bands: tuple[str, ...],
) -> None:
    """Default band order must match the GeoWatch data contract."""
    assert (
        resolve_band_names(
            input_channels=input_channels,
            band_names=None,
        )
        == expected_bands
    )


def build_known_rgb_convolution() -> nn.Conv2d:
    """Create a convolution with identifiable RGB channel values."""
    convolution = nn.Conv2d(
        in_channels=3,
        out_channels=2,
        kernel_size=1,
        bias=False,
    )

    with torch.no_grad():
        convolution.weight[
            :,
            0,
        ].fill_(1.0)  # ImageNet red

        convolution.weight[
            :,
            1,
        ].fill_(2.0)  # ImageNet green

        convolution.weight[
            :,
            2,
        ].fill_(3.0)  # ImageNet blue

    return convolution


def test_four_band_pretrained_mapping() -> None:
    """Four-band adaptation must map RGB weights by spectral identity."""
    original = build_known_rgb_convolution()

    adapted = adapt_first_convolution(
        original_convolution=original,
        input_channels=4,
        band_names=(
            "B02",
            "B03",
            "B04",
            "B08",
        ),
        pretrained=True,
    )

    scale = 3.0 / 4.0

    expected_channel_values = (
        3.0 * scale,  # B02 receives blue
        2.0 * scale,  # B03 receives green
        1.0 * scale,  # B04 receives red
        2.0 * scale,  # B08 receives RGB mean
    )

    assert tuple(adapted.weight.shape) == (
        2,
        4,
        1,
        1,
    )

    for channel_index, expected_value in enumerate(
        expected_channel_values
    ):
        expected = torch.full_like(
            adapted.weight[
                :,
                channel_index,
            ],
            expected_value,
        )

        torch.testing.assert_close(
            adapted.weight[
                :,
                channel_index,
            ],
            expected,
            rtol=0.0,
            atol=1e-7,
        )


def test_six_band_pretrained_mapping() -> None:
    """Additional NIR/SWIR bands must receive the mean RGB filter."""
    original = build_known_rgb_convolution()

    adapted = adapt_first_convolution(
        original_convolution=original,
        input_channels=6,
        band_names=(
            "B02",
            "B03",
            "B04",
            "B08",
            "B11",
            "B12",
        ),
        pretrained=True,
    )

    scale = 3.0 / 6.0

    expected_channel_values = (
        3.0 * scale,
        2.0 * scale,
        1.0 * scale,
        2.0 * scale,
        2.0 * scale,
        2.0 * scale,
    )

    assert tuple(adapted.weight.shape) == (
        2,
        6,
        1,
        1,
    )

    for channel_index, expected_value in enumerate(
        expected_channel_values
    ):
        expected = torch.full_like(
            adapted.weight[
                :,
                channel_index,
            ],
            expected_value,
        )

        torch.testing.assert_close(
            adapted.weight[
                :,
                channel_index,
            ],
            expected,
            rtol=0.0,
            atol=1e-7,
        )


def test_non_pretrained_adaptation_is_valid() -> None:
    """Randomly initialized adaptation must be finite and nonzero."""
    original = build_known_rgb_convolution()

    adapted = adapt_first_convolution(
        original_convolution=original,
        input_channels=4,
        band_names=(
            "B02",
            "B03",
            "B04",
            "B08",
        ),
        pretrained=False,
    )

    assert tuple(adapted.weight.shape) == (
        2,
        4,
        1,
        1,
    )
    assert torch.isfinite(
        adapted.weight
    ).all()
    assert torch.count_nonzero(
        adapted.weight
    ) > 0


def test_missing_rgb_band_is_rejected() -> None:
    """ImageNet adaptation requires the three visible Sentinel bands."""
    with pytest.raises(
        EncoderError,
        match="Missing bands",
    ):
        resolve_band_names(
            input_channels=4,
            band_names=(
                "B02",
                "B03",
                "B08",
                "B11",
            ),
        )
