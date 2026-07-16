"""Multispectral ResNet-18 encoder for GeoWatch.

The encoder exposes intermediate features at five spatial scales for use by
the Siamese U-Net decoder. Torchvision provides the canonical ResNet-18
building blocks, but the multispectral input adaptation and feature-output
contract are implemented explicitly for GeoWatch.

No change-detection library or prebuilt Siamese architecture is used.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


LOGGER = logging.getLogger(
    "geowatch.multispectral_encoder"
)

SUPPORTED_INPUT_CHANNELS = (
    4,
    6,
)

DEFAULT_BANDS_BY_CHANNEL_COUNT: dict[
    int,
    tuple[str, ...],
] = {
    4: (
        "B02",
        "B03",
        "B04",
        "B08",
    ),
    6: (
        "B02",
        "B03",
        "B04",
        "B08",
        "B11",
        "B12",
    ),
}

IMAGENET_RGB_INDEX_BY_SENTINEL_BAND = {
    "B04": 0,
    "B03": 1,
    "B02": 2,
}


class EncoderError(ValueError):
    """Raised when the encoder configuration or input is invalid."""


class EncoderFeatures(NamedTuple):
    """Feature maps produced by the ResNet-18 encoder.

    Attributes:
        stem: Feature map after the initial convolution, normalization and
            activation. Spatial resolution is one-half of the input.
        stage1: ResNet layer-1 output at one-quarter input resolution.
        stage2: ResNet layer-2 output at one-eighth input resolution.
        stage3: ResNet layer-3 output at one-sixteenth input resolution.
        stage4: ResNet layer-4 output at one-thirty-second input resolution.
    """

    stem: Tensor
    stage1: Tensor
    stage2: Tensor
    stage3: Tensor
    stage4: Tensor


def resolve_band_names(
    input_channels: int,
    band_names: Sequence[str] | None,
) -> tuple[str, ...]:
    """Validate and normalize the encoder band configuration.

    Args:
        input_channels: Number of channels expected in each input image.
        band_names: Optional ordered Sentinel-2 band identifiers.

    Returns:
        Normalized ordered band names.

    Raises:
        EncoderError: If the channel count or band configuration is invalid.
    """
    if input_channels not in SUPPORTED_INPUT_CHANNELS:
        raise EncoderError(
            "GeoWatch currently supports only 4-channel and "
            f"6-channel inputs; received {input_channels}."
        )

    if band_names is None:
        return DEFAULT_BANDS_BY_CHANNEL_COUNT[
            input_channels
        ]

    normalized = tuple(
        str(band).strip().upper()
        for band in band_names
    )

    if len(normalized) != input_channels:
        raise EncoderError(
            "The number of band names must equal input_channels. "
            f"Received {len(normalized)} names for "
            f"{input_channels} channels."
        )

    if any(
        not band
        for band in normalized
    ):
        raise EncoderError(
            "Band names must not be empty."
        )

    if len(set(normalized)) != len(normalized):
        raise EncoderError(
            f"Band names must be unique; received {normalized}."
        )

    required_rgb_bands = {
        "B02",
        "B03",
        "B04",
    }

    missing_rgb = sorted(
        required_rgb_bands.difference(
            normalized
        )
    )

    if missing_rgb:
        raise EncoderError(
            "The current ImageNet adaptation requires B02, B03 "
            f"and B04. Missing bands: {missing_rgb}."
        )

    return normalized


def adapt_first_convolution(
    original_convolution: nn.Conv2d,
    input_channels: int,
    band_names: Sequence[str],
    pretrained: bool,
) -> nn.Conv2d:
    """Adapt a three-channel ResNet convolution to multispectral input.

    ImageNet RGB weights are assigned by spectral identity rather than tensor
    position. Sentinel-2 B04 receives the ImageNet red filter, B03 receives
    green and B02 receives blue. Additional spectral channels receive the
    channel-wise mean RGB filter.

    The complete weight tensor is scaled by ``3 / input_channels`` to reduce
    the activation-magnitude shift caused by increasing the input fan-in.

    Args:
        original_convolution: ResNet's original three-channel convolution.
        input_channels: Required multispectral input-channel count.
        band_names: Ordered Sentinel-2 bands matching the input tensor.
        pretrained: Whether the original convolution contains ImageNet
            pretrained weights.

    Returns:
        A convolution compatible with the multispectral input.

    Raises:
        EncoderError: If the original convolution is incompatible.
    """
    if original_convolution.in_channels != 3:
        raise EncoderError(
            "Expected a three-channel torchvision ResNet convolution; "
            f"received {original_convolution.in_channels} channels."
        )

    if len(band_names) != input_channels:
        raise EncoderError(
            "Band-name count and input-channel count do not match."
        )

    adapted = nn.Conv2d(
        in_channels=input_channels,
        out_channels=original_convolution.out_channels,
        kernel_size=original_convolution.kernel_size,
        stride=original_convolution.stride,
        padding=original_convolution.padding,
        dilation=original_convolution.dilation,
        groups=original_convolution.groups,
        bias=False,
        padding_mode=original_convolution.padding_mode,
        device=original_convolution.weight.device,
        dtype=original_convolution.weight.dtype,
    )

    if not pretrained:
        nn.init.kaiming_normal_(
            adapted.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        return adapted

    original_weight = (
        original_convolution.weight.detach()
    )
    mean_rgb_weight = original_weight.mean(
        dim=1
    )

    with torch.no_grad():
        for channel_index, band_name in enumerate(
            band_names
        ):
            imagenet_index = (
                IMAGENET_RGB_INDEX_BY_SENTINEL_BAND.get(
                    band_name
                )
            )

            if imagenet_index is None:
                adapted.weight[
                    :,
                    channel_index,
                ].copy_(
                    mean_rgb_weight
                )
            else:
                adapted.weight[
                    :,
                    channel_index,
                ].copy_(
                    original_weight[
                        :,
                        imagenet_index,
                    ]
                )

        adapted.weight.mul_(
            3.0 / float(input_channels)
        )

    return adapted


class MultispectralResNet18Encoder(nn.Module):
    """ResNet-18 feature encoder adapted for Sentinel-2 inputs.

    The classification pooling and fully connected layers are intentionally
    omitted. The module returns five feature maps for later Siamese
    difference fusion and U-Net decoding.
    """

    FEATURE_CHANNELS = (
        64,
        64,
        128,
        256,
        512,
    )

    FEATURE_DOWNSAMPLE_FACTORS = (
        2,
        4,
        8,
        16,
        32,
    )

    def __init__(
        self,
        input_channels: int = 4,
        band_names: Sequence[str] | None = None,
        pretrained: bool = True,
    ) -> None:
        """Initialize the multispectral encoder.

        Args:
            input_channels: Number of input Sentinel-2 bands.
            band_names: Ordered band names matching the input tensor.
            pretrained: Load ImageNet ResNet-18 weights before adapting the
                first convolution.

        Raises:
            EncoderError: If the channel or band configuration is invalid.
        """
        super().__init__()

        resolved_bands = resolve_band_names(
            input_channels=input_channels,
            band_names=band_names,
        )

        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = resnet18(
            weights=weights
        )

        backbone.conv1 = adapt_first_convolution(
            original_convolution=backbone.conv1,
            input_channels=input_channels,
            band_names=resolved_bands,
            pretrained=pretrained,
        )

        self.input_channels = input_channels
        self.band_names = resolved_bands
        self.pretrained = pretrained

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    @property
    def feature_channels(
        self,
    ) -> tuple[int, ...]:
        """Return the channel count at every emitted feature scale."""
        return self.FEATURE_CHANNELS

    @property
    def feature_downsample_factors(
        self,
    ) -> tuple[int, ...]:
        """Return the input downsampling factor for each feature map."""
        return self.FEATURE_DOWNSAMPLE_FACTORS

    def validate_input(
        self,
        image: Tensor,
    ) -> None:
        """Validate one multispectral image tensor.

        Args:
            image: Tensor shaped ``[batch, channels, height, width]``.

        Raises:
            EncoderError: If the tensor violates the architecture contract.
        """
        if image.ndim != 4:
            raise EncoderError(
                "Encoder input must have shape "
                "[batch, channels, height, width]; "
                f"received {tuple(image.shape)}."
            )

        if image.shape[1] != self.input_channels:
            raise EncoderError(
                f"Encoder expects {self.input_channels} channels "
                f"{self.band_names}; received {image.shape[1]}."
            )

        if not torch.is_floating_point(
            image
        ):
            raise EncoderError(
                "Encoder input must use a floating-point data type; "
                f"received {image.dtype}."
            )

        height = int(
            image.shape[-2]
        )
        width = int(
            image.shape[-1]
        )

        if height < 32 or width < 32:
            raise EncoderError(
                "Input height and width must each be at least 32; "
                f"received {height}x{width}."
            )

        if height % 32 != 0 or width % 32 != 0:
            raise EncoderError(
                "Week 3 inputs must have spatial dimensions divisible "
                f"by 32; received {height}x{width}."
            )

    def forward(
        self,
        image: Tensor,
    ) -> EncoderFeatures:
        """Extract five hierarchical feature maps.

        Args:
            image: Multispectral tensor shaped
                ``[batch, channels, height, width]``.

        Returns:
            Encoder feature maps from one-half through one-thirty-second
            input resolution.
        """
        self.validate_input(
            image
        )

        stem = self.relu(
            self.bn1(
                self.conv1(
                    image
                )
            )
        )

        features = self.maxpool(
            stem
        )

        stage1 = self.layer1(
            features
        )
        stage2 = self.layer2(
            stage1
        )
        stage3 = self.layer3(
            stage2
        )
        stage4 = self.layer4(
            stage3
        )

        return EncoderFeatures(
            stem=stem,
            stage1=stage1,
            stage2=stage2,
            stage3=stage3,
            stage4=stage4,
        )


def expected_feature_shapes(
    batch_size: int,
    height: int,
    width: int,
) -> tuple[tuple[int, ...], ...]:
    """Return the expected ResNet-18 feature shapes."""
    return tuple(
        (
            batch_size,
            channels,
            height // factor,
            width // factor,
        )
        for channels, factor in zip(
            MultispectralResNet18Encoder.FEATURE_CHANNELS,
            MultispectralResNet18Encoder.FEATURE_DOWNSAMPLE_FACTORS,
            strict=True,
        )
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the synthetic encoder-audit CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a synthetic shape audit for the GeoWatch "
            "multispectral ResNet-18 encoder."
        )
    )

    parser.add_argument(
        "--input-channels",
        type=int,
        choices=SUPPORTED_INPUT_CHANNELS,
        default=4,
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        default=None,
        help=(
            "Optional ordered Sentinel-2 band names. Defaults to "
            "the standard GeoWatch 4-band or 6-band configuration."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help=(
            "Download and use ImageNet weights. Omit this flag for "
            "offline synthetic architecture checks."
        ),
    )
    parser.add_argument(
        "--device",
        choices=(
            "cpu",
            "cuda",
        ),
        default="cpu",
    )
    parser.add_argument(
        "--log-level",
        choices=(
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ),
        default="INFO",
    )

    return parser


def main() -> int:
    """Execute the synthetic encoder shape audit."""
    parser = build_argument_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(
            logging,
            args.log_level,
        ),
        format="%(levelname)s: %(message)s",
    )

    try:
        if args.batch_size <= 0:
            raise EncoderError(
                "batch-size must be greater than zero."
            )

        if (
            args.device == "cuda"
            and not torch.cuda.is_available()
        ):
            raise EncoderError(
                "CUDA was requested but is unavailable."
            )

        device = torch.device(
            args.device
        )

        encoder = MultispectralResNet18Encoder(
            input_channels=args.input_channels,
            band_names=args.bands,
            pretrained=args.pretrained,
        ).to(
            device
        )

        encoder.eval()

        image = torch.randn(
            args.batch_size,
            args.input_channels,
            args.height,
            args.width,
            device=device,
        )

        with torch.inference_mode():
            features = encoder(
                image
            )

        actual_shapes = tuple(
            tuple(feature.shape)
            for feature in features
        )
        expected_shapes = expected_feature_shapes(
            batch_size=args.batch_size,
            height=args.height,
            width=args.width,
        )

        if actual_shapes != expected_shapes:
            raise RuntimeError(
                "Unexpected encoder feature shapes. "
                f"Expected {expected_shapes}, received {actual_shapes}."
            )

        total_parameters = sum(
            parameter.numel()
            for parameter in encoder.parameters()
        )

        trainable_parameters = sum(
            parameter.numel()
            for parameter in encoder.parameters()
            if parameter.requires_grad
        )

        print(
            "GeoWatch multispectral encoder audit passed"
        )
        print(
            "  Input shape:",
            tuple(image.shape),
        )
        print(
            "  Bands:",
            encoder.band_names,
        )
        print(
            "  Pretrained:",
            encoder.pretrained,
        )

        for feature_name, feature in zip(
            EncoderFeatures._fields,
            features,
            strict=True,
        ):
            print(
                f"  {feature_name}:",
                tuple(feature.shape),
            )

        print(
            "  Total parameters:",
            total_parameters,
        )
        print(
            "  Trainable parameters:",
            trainable_parameters,
        )

        return 0

    except (
        EncoderError,
        RuntimeError,
        ValueError,
        TypeError,
    ) as error:
        LOGGER.error(
            "%s",
            error,
        )
        return 1

    except Exception:
        LOGGER.exception(
            "Unexpected encoder-audit failure."
        )
        return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )
