"""Weight-shared Siamese U-Net for GeoWatch change detection.

Two bi-temporal multispectral images are encoded by the same ResNet-18
encoder. Corresponding hierarchical feature maps are fused using absolute
difference and decoded into one full-resolution binary change logit map.

The module returns raw logits. Sigmoid conversion belongs in evaluation or
inference code, not inside the architecture.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from typing import NamedTuple

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from src.models.encoder import (
    DEFAULT_BANDS_BY_CHANNEL_COUNT,
    EncoderFeatures,
    MultispectralResNet18Encoder,
    SUPPORTED_INPUT_CHANNELS,
)


LOGGER = logging.getLogger(
    "geowatch.siamese_unet"
)


class ModelError(ValueError):
    """Raised when the Siamese U-Net configuration or input is invalid."""


class FusedFeatures(NamedTuple):
    """Absolute-difference feature maps at five encoder scales."""

    stem: Tensor
    stage1: Tensor
    stage2: Tensor
    stage3: Tensor
    stage4: Tensor


def resolve_group_count(
    channels: int,
    preferred_groups: int,
) -> int:
    """Select the largest valid GroupNorm divisor.

    Args:
        channels: Number of channels being normalized.
        preferred_groups: Preferred maximum number of groups.

    Returns:
        A positive group count that divides ``channels``.

    Raises:
        ModelError: If either argument is invalid.
    """
    if channels <= 0:
        raise ModelError(
            f"channels must be positive; received {channels}."
        )

    if preferred_groups <= 0:
        raise ModelError(
            "preferred_groups must be greater than zero."
        )

    maximum_groups = min(
        channels,
        preferred_groups,
    )

    for groups in range(
        maximum_groups,
        0,
        -1,
    ):
        if channels % groups == 0:
            return groups

    raise ModelError(
        f"Could not select GroupNorm groups for {channels} channels."
    )


class ConvNormActivation(nn.Sequential):
    """Convolution followed by GroupNorm and ReLU."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        preferred_norm_groups: int = 8,
    ) -> None:
        """Initialize one decoder convolution block."""
        if input_channels <= 0 or output_channels <= 0:
            raise ModelError(
                "Convolution channel counts must be positive."
            )

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ModelError(
                "kernel_size must be a positive odd integer."
            )

        padding = kernel_size // 2
        group_count = resolve_group_count(
            channels=output_channels,
            preferred_groups=preferred_norm_groups,
        )

        super().__init__(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=group_count,
                num_channels=output_channels,
            ),
            nn.ReLU(
                inplace=True,
            ),
        )


class DoubleConv(nn.Module):
    """Two decoder convolutions with optional spatial dropout."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        preferred_norm_groups: int = 8,
        dropout_probability: float = 0.0,
    ) -> None:
        """Initialize the double-convolution block."""
        super().__init__()

        if not 0.0 <= dropout_probability < 1.0:
            raise ModelError(
                "dropout_probability must be in [0, 1)."
            )

        self.first = ConvNormActivation(
            input_channels=input_channels,
            output_channels=output_channels,
            preferred_norm_groups=preferred_norm_groups,
        )
        self.dropout = (
            nn.Dropout2d(
                p=dropout_probability
            )
            if dropout_probability > 0.0
            else nn.Identity()
        )
        self.second = ConvNormActivation(
            input_channels=output_channels,
            output_channels=output_channels,
            preferred_norm_groups=preferred_norm_groups,
        )

    def forward(
        self,
        features: Tensor,
    ) -> Tensor:
        """Refine one decoder feature map."""
        features = self.first(
            features
        )
        features = self.dropout(
            features
        )
        return self.second(
            features
        )


class DecoderBlock(nn.Module):
    """Upsample, concatenate a fused skip feature, and refine."""

    def __init__(
        self,
        input_channels: int,
        skip_channels: int,
        output_channels: int,
        preferred_norm_groups: int = 8,
        dropout_probability: float = 0.0,
    ) -> None:
        """Initialize one U-Net decoder level."""
        super().__init__()

        if min(
            input_channels,
            skip_channels,
            output_channels,
        ) <= 0:
            raise ModelError(
                "Decoder channel counts must all be positive."
            )

        self.upsample_projection = ConvNormActivation(
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=1,
            preferred_norm_groups=preferred_norm_groups,
        )

        self.refinement = DoubleConv(
            input_channels=(
                output_channels
                + skip_channels
            ),
            output_channels=output_channels,
            preferred_norm_groups=preferred_norm_groups,
            dropout_probability=dropout_probability,
        )

    def forward(
        self,
        decoder_features: Tensor,
        skip_features: Tensor,
    ) -> Tensor:
        """Decode one level using its matching fused skip feature."""
        if decoder_features.ndim != 4:
            raise ModelError(
                "Decoder features must be four-dimensional."
            )

        if skip_features.ndim != 4:
            raise ModelError(
                "Skip features must be four-dimensional."
            )

        if (
            decoder_features.shape[0]
            != skip_features.shape[0]
        ):
            raise ModelError(
                "Decoder and skip batch sizes do not match."
            )

        decoder_features = functional.interpolate(
            decoder_features,
            size=skip_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        decoder_features = self.upsample_projection(
            decoder_features
        )

        concatenated = torch.cat(
            (
                decoder_features,
                skip_features,
            ),
            dim=1,
        )

        return self.refinement(
            concatenated
        )


class SegmentationHead(nn.Module):
    """Restore full resolution and emit one binary-change logit."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        preferred_norm_groups: int = 8,
        dropout_probability: float = 0.0,
    ) -> None:
        """Initialize the full-resolution segmentation head."""
        super().__init__()

        if hidden_channels <= 0:
            raise ModelError(
                "hidden_channels must be positive."
            )

        self.refinement = DoubleConv(
            input_channels=input_channels,
            output_channels=hidden_channels,
            preferred_norm_groups=preferred_norm_groups,
            dropout_probability=dropout_probability,
        )

        self.classifier = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=1,
            kernel_size=1,
            bias=True,
        )

    def forward(
        self,
        decoder_features: Tensor,
        output_size: tuple[int, int],
    ) -> Tensor:
        """Produce full-resolution raw change logits."""
        decoder_features = functional.interpolate(
            decoder_features,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        decoder_features = self.refinement(
            decoder_features
        )

        return self.classifier(
            decoder_features
        )


def fuse_absolute_difference(
    before_features: EncoderFeatures,
    after_features: EncoderFeatures,
) -> FusedFeatures:
    """Fuse corresponding feature maps using absolute difference."""
    fused: list[Tensor] = []

    for before_feature, after_feature in zip(
        before_features,
        after_features,
        strict=True,
    ):
        if before_feature.shape != after_feature.shape:
            raise ModelError(
                "Corresponding encoder features have different shapes: "
                f"{tuple(before_feature.shape)} versus "
                f"{tuple(after_feature.shape)}."
            )

        fused.append(
            torch.abs(
                after_feature
                - before_feature
            )
        )

    return FusedFeatures(
        *fused
    )


class SiameseUNet(nn.Module):
    """Weight-shared Siamese ResNet-18 U-Net.

    The same encoder instance processes both temporal observations. Absolute
    differences are computed at all encoder scales and supplied to the
    decoder as its bottleneck and skip features.
    """

    def __init__(
        self,
        input_channels: int = 4,
        band_names: Sequence[str] | None = None,
        pretrained_encoder: bool = True,
        decoder_channels: Sequence[int] = (
            256,
            128,
            64,
            64,
        ),
        head_channels: int = 32,
        preferred_norm_groups: int = 8,
        dropout_probability: float = 0.0,
    ) -> None:
        """Initialize the GeoWatch Siamese U-Net.

        Args:
            input_channels: Number of channels in each temporal image.
            band_names: Ordered Sentinel-2 bands matching the input tensors.
            pretrained_encoder: Initialize the shared ResNet encoder from
                ImageNet weights.
            decoder_channels: Output channels for the four decoder levels,
                ordered from one-sixteenth to one-half resolution.
            head_channels: Channels used by the full-resolution head.
            preferred_norm_groups: Preferred maximum GroupNorm groups.
            dropout_probability: Decoder spatial-dropout probability.

        Raises:
            ModelError: If decoder or head configuration is invalid.
        """
        super().__init__()

        normalized_decoder_channels = tuple(
            int(channels)
            for channels in decoder_channels
        )

        if len(normalized_decoder_channels) != 4:
            raise ModelError(
                "decoder_channels must contain exactly four values."
            )

        if any(
            channels <= 0
            for channels in normalized_decoder_channels
        ):
            raise ModelError(
                "Every decoder channel count must be positive."
            )

        if head_channels <= 0:
            raise ModelError(
                "head_channels must be positive."
            )

        self.encoder = MultispectralResNet18Encoder(
            input_channels=input_channels,
            band_names=band_names,
            pretrained=pretrained_encoder,
        )

        encoder_channels = self.encoder.feature_channels

        stage3_channels = encoder_channels[3]
        stage2_channels = encoder_channels[2]
        stage1_channels = encoder_channels[1]
        stem_channels = encoder_channels[0]
        deepest_channels = encoder_channels[4]

        decoder_stage3_channels = (
            normalized_decoder_channels[0]
        )
        decoder_stage2_channels = (
            normalized_decoder_channels[1]
        )
        decoder_stage1_channels = (
            normalized_decoder_channels[2]
        )
        decoder_stem_channels = (
            normalized_decoder_channels[3]
        )

        self.decoder_stage3 = DecoderBlock(
            input_channels=deepest_channels,
            skip_channels=stage3_channels,
            output_channels=decoder_stage3_channels,
            preferred_norm_groups=preferred_norm_groups,
            dropout_probability=dropout_probability,
        )
        self.decoder_stage2 = DecoderBlock(
            input_channels=decoder_stage3_channels,
            skip_channels=stage2_channels,
            output_channels=decoder_stage2_channels,
            preferred_norm_groups=preferred_norm_groups,
            dropout_probability=dropout_probability,
        )
        self.decoder_stage1 = DecoderBlock(
            input_channels=decoder_stage2_channels,
            skip_channels=stage1_channels,
            output_channels=decoder_stage1_channels,
            preferred_norm_groups=preferred_norm_groups,
            dropout_probability=dropout_probability,
        )
        self.decoder_stem = DecoderBlock(
            input_channels=decoder_stage1_channels,
            skip_channels=stem_channels,
            output_channels=decoder_stem_channels,
            preferred_norm_groups=preferred_norm_groups,
            dropout_probability=dropout_probability,
        )

        self.segmentation_head = SegmentationHead(
            input_channels=decoder_stem_channels,
            hidden_channels=head_channels,
            preferred_norm_groups=preferred_norm_groups,
            dropout_probability=dropout_probability,
        )

        self.input_channels = input_channels
        self.band_names = self.encoder.band_names
        self.pretrained_encoder = pretrained_encoder
        self.decoder_channels = normalized_decoder_channels
        self.head_channels = head_channels

    def validate_pair(
        self,
        before: Tensor,
        after: Tensor,
    ) -> None:
        """Validate a bi-temporal input pair."""
        if before.shape != after.shape:
            raise ModelError(
                "Before and after tensors must have identical shapes; "
                f"received {tuple(before.shape)} and "
                f"{tuple(after.shape)}."
            )

        if before.ndim != 4:
            raise ModelError(
                "Inputs must have shape "
                "[batch, channels, height, width]."
            )

        if before.device != after.device:
            raise ModelError(
                "Before and after tensors must be on the same device."
            )

        if before.dtype != after.dtype:
            raise ModelError(
                "Before and after tensors must use the same dtype."
            )

        if not torch.is_floating_point(
            before
        ):
            raise ModelError(
                "Siamese U-Net inputs must be floating-point tensors."
            )

        if before.shape[1] != self.input_channels:
            raise ModelError(
                f"Model expects {self.input_channels} channels "
                f"{self.band_names}; received {before.shape[1]}."
            )

        height = int(
            before.shape[-2]
        )
        width = int(
            before.shape[-1]
        )

        if height < 32 or width < 32:
            raise ModelError(
                "Input height and width must each be at least 32."
            )

        if height % 32 != 0 or width % 32 != 0:
            raise ModelError(
                "Input height and width must be divisible by 32; "
                f"received {height}x{width}."
            )

    def encode_pair(
        self,
        before: Tensor,
        after: Tensor,
    ) -> tuple[
        EncoderFeatures,
        EncoderFeatures,
    ]:
        """Encode both dates using the same encoder parameters."""
        before_features = self.encoder(
            before
        )
        after_features = self.encoder(
            after
        )

        return (
            before_features,
            after_features,
        )

    def forward(
        self,
        before: Tensor,
        after: Tensor,
    ) -> Tensor:
        """Predict one raw binary-change logit for every input pixel."""
        self.validate_pair(
            before,
            after,
        )

        (
            before_features,
            after_features,
        ) = self.encode_pair(
            before,
            after,
        )

        fused = fuse_absolute_difference(
            before_features=before_features,
            after_features=after_features,
        )

        decoder = self.decoder_stage3(
            decoder_features=fused.stage4,
            skip_features=fused.stage3,
        )
        decoder = self.decoder_stage2(
            decoder_features=decoder,
            skip_features=fused.stage2,
        )
        decoder = self.decoder_stage1(
            decoder_features=decoder,
            skip_features=fused.stage1,
        )
        decoder = self.decoder_stem(
            decoder_features=decoder,
            skip_features=fused.stem,
        )

        logits = self.segmentation_head(
            decoder_features=decoder,
            output_size=(
                int(before.shape[-2]),
                int(before.shape[-1]),
            ),
        )

        expected_shape = (
            int(before.shape[0]),
            1,
            int(before.shape[-2]),
            int(before.shape[-1]),
        )

        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected Siamese U-Net output shape. "
                f"Expected {expected_shape}, "
                f"received {tuple(logits.shape)}."
            )

        return logits


def count_parameters(
    model: nn.Module,
) -> tuple[int, int]:
    """Return total and trainable model parameter counts."""
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the synthetic Siamese U-Net audit CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a synthetic shape and symmetry audit for the "
            "GeoWatch Siamese U-Net."
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
        "--decoder-channels",
        nargs=4,
        type=int,
        default=[
            256,
            128,
            64,
            64,
        ],
    )
    parser.add_argument(
        "--head-channels",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--norm-groups",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
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
        "--seed",
        type=int,
        default=42,
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
    """Run the synthetic architecture audit."""
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
            raise ModelError(
                "batch-size must be greater than zero."
            )

        if (
            args.device == "cuda"
            and not torch.cuda.is_available()
        ):
            raise ModelError(
                "CUDA was requested but is unavailable."
            )

        torch.manual_seed(
            args.seed
        )

        device = torch.device(
            args.device
        )

        model = SiameseUNet(
            input_channels=args.input_channels,
            band_names=args.bands,
            pretrained_encoder=args.pretrained,
            decoder_channels=args.decoder_channels,
            head_channels=args.head_channels,
            preferred_norm_groups=args.norm_groups,
            dropout_probability=args.dropout,
        ).to(
            device
        )

        model.eval()

        before = torch.randn(
            args.batch_size,
            args.input_channels,
            args.height,
            args.width,
            device=device,
        )
        after = torch.randn(
            args.batch_size,
            args.input_channels,
            args.height,
            args.width,
            device=device,
        )

        with torch.inference_mode():
            logits_forward = model(
                before,
                after,
            )
            logits_swapped = model(
                after,
                before,
            )

        maximum_swap_error = float(
            torch.max(
                torch.abs(
                    logits_forward
                    - logits_swapped
                )
            ).item()
        )

        if not torch.allclose(
            logits_forward,
            logits_swapped,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise RuntimeError(
                "Absolute-difference Siamese model is not "
                "date-order symmetric."
            )

        total_parameters, trainable_parameters = (
            count_parameters(
                model
            )
        )

        state_keys = tuple(
            model.state_dict().keys()
        )

        if not any(
            key.startswith("encoder.")
            for key in state_keys
        ):
            raise RuntimeError(
                "Shared encoder parameters were not found."
            )

        if any(
            key.startswith("encoder_before.")
            or key.startswith("encoder_after.")
            for key in state_keys
        ):
            raise RuntimeError(
                "Separate temporal encoder parameters were detected."
            )

        print(
            "GeoWatch Siamese U-Net audit passed"
        )
        print(
            "  Before shape:",
            tuple(before.shape),
        )
        print(
            "  After shape:",
            tuple(after.shape),
        )
        print(
            "  Logit shape:",
            tuple(logits_forward.shape),
        )
        print(
            "  Bands:",
            model.band_names,
        )
        print(
            "  Decoder channels:",
            model.decoder_channels,
        )
        print(
            "  Pretrained encoder:",
            model.pretrained_encoder,
        )
        print(
            "  Shared encoder object:",
            True,
        )
        print(
            "  Separate temporal encoders:",
            False,
        )
        print(
            "  Date-order symmetric:",
            True,
        )
        print(
            "  Maximum swap error:",
            maximum_swap_error,
        )
        print(
            "  Raw logits returned:",
            True,
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
        ModelError,
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
            "Unexpected Siamese U-Net audit failure."
        )
        return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )
