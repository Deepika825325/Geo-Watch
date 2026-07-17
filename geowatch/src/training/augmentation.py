"""Paired geometric augmentation for multispectral change detection.

The same spatial transformation is applied to the before image, after image
and binary change mask. Only exact geometric operations are supported:

* horizontal flip
* vertical flip
* rotations in multiples of 90 degrees

No interpolation is used, so binary masks remain binary and Sentinel-2
reflectance values remain unchanged. Spectral or colour-based augmentation is
deliberately rejected because arbitrary colour jitter would corrupt the
physical meaning of multispectral bands.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor


LOGGER = logging.getLogger(
    "geowatch.augmentation"
)


class AugmentationConfigurationError(ValueError):
    """Raised when augmentation configuration or tensors are invalid."""


@dataclass(frozen=True)
class GeometricDecision:
    """One sampled transformation shared by a temporal image pair."""

    horizontal_flip: bool
    vertical_flip: bool
    quarter_turns: int

    def __post_init__(
        self,
    ) -> None:
        """Validate the number of counter-clockwise quarter turns."""
        if self.quarter_turns not in {
            0,
            1,
            2,
            3,
        }:
            raise AugmentationConfigurationError(
                "quarter_turns must be one of 0, 1, 2 or 3."
            )


def validate_probability(
    value: float,
    name: str,
) -> float:
    """Validate an augmentation probability."""
    probability = float(
        value
    )

    if not 0.0 <= probability <= 1.0:
        raise AugmentationConfigurationError(
            f"{name} must be between 0 and 1; received {probability}."
        )

    return probability


def validate_paired_tensors(
    before: Tensor,
    after: Tensor,
    mask: Tensor,
) -> None:
    """Validate a channel-first bi-temporal sample and binary mask."""
    if before.ndim != 3:
        raise AugmentationConfigurationError(
            "Before image must have shape [C, H, W]; "
            f"received {tuple(before.shape)}."
        )

    if after.ndim != 3:
        raise AugmentationConfigurationError(
            "After image must have shape [C, H, W]; "
            f"received {tuple(after.shape)}."
        )

    if mask.ndim != 3:
        raise AugmentationConfigurationError(
            "Mask must have shape [1, H, W]; "
            f"received {tuple(mask.shape)}."
        )

    if before.shape != after.shape:
        raise AugmentationConfigurationError(
            "Before and after images must have identical shapes; "
            f"received {tuple(before.shape)} and {tuple(after.shape)}."
        )

    if mask.shape[0] != 1:
        raise AugmentationConfigurationError(
            "Binary mask must contain exactly one channel."
        )

    if before.shape[-2:] != mask.shape[-2:]:
        raise AugmentationConfigurationError(
            "Image and mask spatial dimensions must match; "
            f"received {tuple(before.shape[-2:])} and "
            f"{tuple(mask.shape[-2:])}."
        )

    if not before.is_floating_point():
        raise AugmentationConfigurationError(
            "Before image must be floating-point."
        )

    if not after.is_floating_point():
        raise AugmentationConfigurationError(
            "After image must be floating-point."
        )

    if not mask.is_floating_point():
        raise AugmentationConfigurationError(
            "Mask must be floating-point."
        )

    if before.dtype != after.dtype:
        raise AugmentationConfigurationError(
            "Before and after tensors must use the same dtype."
        )

    if before.device != after.device or before.device != mask.device:
        raise AugmentationConfigurationError(
            "Before, after and mask tensors must use the same device."
        )

    for tensor, tensor_name in (
        (
            before,
            "Before image",
        ),
        (
            after,
            "After image",
        ),
        (
            mask,
            "Mask",
        ),
    ):
        if not bool(
            torch.isfinite(
                tensor
            ).all().item()
        ):
            raise AugmentationConfigurationError(
                f"{tensor_name} contains non-finite values."
            )

    valid_mask_values = torch.logical_or(
        mask == 0,
        mask == 1,
    )

    if not bool(
        valid_mask_values.all().item()
    ):
        raise AugmentationConfigurationError(
            "Mask must contain only binary values 0 and 1."
        )


def apply_geometric_decision(
    tensor: Tensor,
    decision: GeometricDecision,
) -> Tensor:
    """Apply one exact geometric decision to a channel-first tensor."""
    transformed = tensor

    if decision.horizontal_flip:
        transformed = torch.flip(
            transformed,
            dims=(
                -1,
            ),
        )

    if decision.vertical_flip:
        transformed = torch.flip(
            transformed,
            dims=(
                -2,
            ),
        )

    if decision.quarter_turns:
        transformed = torch.rot90(
            transformed,
            k=decision.quarter_turns,
            dims=(
                -2,
                -1,
            ),
        )

    return transformed.contiguous()


def apply_paired_geometric_decision(
    before: Tensor,
    after: Tensor,
    mask: Tensor,
    decision: GeometricDecision,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
]:
    """Apply exactly the same transformation to images and mask."""
    validate_paired_tensors(
        before=before,
        after=after,
        mask=mask,
    )

    height = int(
        before.shape[-2]
    )
    width = int(
        before.shape[-1]
    )

    if (
        decision.quarter_turns
        in {
            1,
            3,
        }
        and height != width
    ):
        raise AugmentationConfigurationError(
            "Odd 90-degree rotations require square patches to preserve "
            f"the configured shape; received {(height, width)}."
        )

    return (
        apply_geometric_decision(
            tensor=before,
            decision=decision,
        ),
        apply_geometric_decision(
            tensor=after,
            decision=decision,
        ),
        apply_geometric_decision(
            tensor=mask,
            decision=decision,
        ),
    )


class PairedGeometricAugmentation:
    """Random geometric augmentation shared across a temporal sample.

    Random values are drawn from PyTorch rather than Python's ``random``
    module. This allows the future DataLoader to control reproducibility
    through PyTorch worker seeds.
    """

    def __init__(
        self,
        horizontal_flip_probability: float = 0.5,
        vertical_flip_probability: float = 0.5,
        rotate_90_probability: float = 0.5,
        enabled: bool = True,
    ) -> None:
        """Initialize exact multispectral-safe augmentations."""
        self.enabled = bool(
            enabled
        )

        self.horizontal_flip_probability = validate_probability(
            value=horizontal_flip_probability,
            name="horizontal_flip_probability",
        )
        self.vertical_flip_probability = validate_probability(
            value=vertical_flip_probability,
            name="vertical_flip_probability",
        )
        self.rotate_90_probability = validate_probability(
            value=rotate_90_probability,
            name="rotate_90_probability",
        )

    @staticmethod
    def _sample_event(
        probability: float,
        generator: torch.Generator | None,
    ) -> bool:
        """Sample one Bernoulli event using the PyTorch RNG."""
        if probability <= 0.0:
            return False

        if probability >= 1.0:
            return True

        random_value = torch.rand(
            (),
            generator=generator,
        )

        return bool(
            random_value.item()
            < probability
        )

    def sample_decision(
        self,
        generator: torch.Generator | None = None,
    ) -> GeometricDecision:
        """Sample one transformation decision."""
        if not self.enabled:
            return GeometricDecision(
                horizontal_flip=False,
                vertical_flip=False,
                quarter_turns=0,
            )

        horizontal_flip = self._sample_event(
            probability=self.horizontal_flip_probability,
            generator=generator,
        )
        vertical_flip = self._sample_event(
            probability=self.vertical_flip_probability,
            generator=generator,
        )

        rotate = self._sample_event(
            probability=self.rotate_90_probability,
            generator=generator,
        )

        if rotate:
            quarter_turns = int(
                torch.randint(
                    low=1,
                    high=4,
                    size=(
                        1,
                    ),
                    generator=generator,
                ).item()
            )
        else:
            quarter_turns = 0

        return GeometricDecision(
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            quarter_turns=quarter_turns,
        )

    def apply(
        self,
        before: Tensor,
        after: Tensor,
        mask: Tensor,
        decision: GeometricDecision,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
    ]:
        """Apply an explicit decision to a temporal sample."""
        return apply_paired_geometric_decision(
            before=before,
            after=after,
            mask=mask,
            decision=decision,
        )

    def __call__(
        self,
        before: Tensor,
        after: Tensor,
        mask: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
    ]:
        """Sample and apply one shared geometric transformation."""
        decision = self.sample_decision()

        return self.apply(
            before=before,
            after=after,
            mask=mask,
            decision=decision,
        )


def require_mapping(
    value: Any,
    name: str,
) -> Mapping[str, Any]:
    """Validate and return a configuration mapping."""
    if not isinstance(
        value,
        Mapping,
    ):
        raise AugmentationConfigurationError(
            f"{name} must be a configuration mapping."
        )

    return value


def build_augmentation_from_config(
    config: Mapping[str, Any],
) -> PairedGeometricAugmentation:
    """Build paired geometric augmentation from the training YAML."""
    augmentation_value = config.get(
        "augmentation",
        config,
    )
    augmentation_config = require_mapping(
        augmentation_value,
        "augmentation",
    )

    forbidden_augmentations = (
        "spectral_augmentation_enabled",
        "color_jitter_enabled",
        "brightness_adjustment_enabled",
        "contrast_adjustment_enabled",
    )

    enabled_forbidden = [
        name
        for name in forbidden_augmentations
        if bool(
            augmentation_config.get(
                name,
                False,
            )
        )
    ]

    if enabled_forbidden:
        raise AugmentationConfigurationError(
            "GeoWatch multispectral training forbids reflectance-changing "
            f"augmentations: {enabled_forbidden}."
        )

    return PairedGeometricAugmentation(
        enabled=bool(
            augmentation_config.get(
                "enabled",
                True,
            )
        ),
        horizontal_flip_probability=float(
            augmentation_config[
                "horizontal_flip_probability"
            ]
        ),
        vertical_flip_probability=float(
            augmentation_config[
                "vertical_flip_probability"
            ]
        ),
        rotate_90_probability=float(
            augmentation_config[
                "rotate_90_probability"
            ]
        ),
    )


def load_yaml_config(
    path: Path,
) -> Mapping[str, Any]:
    """Load one UTF-8 YAML training configuration."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Training configuration does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as config_file:
        config = yaml.safe_load(
            config_file
        )

    return require_mapping(
        config,
        "root configuration",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the augmentation audit CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Audit paired geometric augmentation using deterministic "
            "synthetic multispectral tensors."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/train_config.yaml"
        ),
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
    """Run a deterministic paired-augmentation audit."""
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
        config = load_yaml_config(
            args.config
        )
        augmentation = build_augmentation_from_config(
            config
        )

        before = torch.arange(
            4 * 8 * 8,
            dtype=torch.float32,
        ).reshape(
            4,
            8,
            8,
        )

        after = before + 1_000.0

        mask = torch.zeros(
            1,
            8,
            8,
            dtype=torch.float32,
        )
        mask[
            :,
            2:6,
            1:5,
        ] = 1.0

        generator = torch.Generator().manual_seed(
            args.seed
        )

        decision = augmentation.sample_decision(
            generator=generator
        )

        transformed_before, transformed_after, transformed_mask = (
            augmentation.apply(
                before=before,
                after=after,
                mask=mask,
                decision=decision,
            )
        )

        expected_before = apply_geometric_decision(
            tensor=before,
            decision=decision,
        )
        expected_after = apply_geometric_decision(
            tensor=after,
            decision=decision,
        )
        expected_mask = apply_geometric_decision(
            tensor=mask,
            decision=decision,
        )

        if not torch.equal(
            transformed_before,
            expected_before,
        ):
            raise AugmentationConfigurationError(
                "Before image did not receive the sampled transformation."
            )

        if not torch.equal(
            transformed_after,
            expected_after,
        ):
            raise AugmentationConfigurationError(
                "After image did not receive the sampled transformation."
            )

        if not torch.equal(
            transformed_mask,
            expected_mask,
        ):
            raise AugmentationConfigurationError(
                "Mask did not receive the sampled transformation."
            )

        temporal_difference = (
            transformed_after
            - transformed_before
        )

        if not torch.all(
            temporal_difference
            == 1_000.0
        ):
            raise AugmentationConfigurationError(
                "Paired augmentation corrupted temporal alignment."
            )

        mask_values = set(
            torch.unique(
                transformed_mask
            ).tolist()
        )

        if not mask_values.issubset(
            {
                0.0,
                1.0,
            }
        ):
            raise AugmentationConfigurationError(
                f"Augmentation corrupted binary labels: {mask_values}"
            )

        original_values = torch.sort(
            before.flatten()
        ).values
        transformed_values = torch.sort(
            transformed_before.flatten()
        ).values

        if not torch.equal(
            original_values,
            transformed_values,
        ):
            raise AugmentationConfigurationError(
                "Geometric augmentation changed reflectance values."
            )

        print(
            "GeoWatch paired geometric augmentation audit passed"
        )
        print(
            "  Configuration:",
            args.config,
        )
        print(
            "  Enabled:",
            augmentation.enabled,
        )
        print(
            "  Horizontal flip probability:",
            augmentation.horizontal_flip_probability,
        )
        print(
            "  Vertical flip probability:",
            augmentation.vertical_flip_probability,
        )
        print(
            "  Rotate-90 probability:",
            augmentation.rotate_90_probability,
        )
        print(
            "  Sampled horizontal flip:",
            decision.horizontal_flip,
        )
        print(
            "  Sampled vertical flip:",
            decision.vertical_flip,
        )
        print(
            "  Sampled quarter turns:",
            decision.quarter_turns,
        )
        print(
            "  Temporal alignment preserved:",
            True,
        )
        print(
            "  Reflectance values preserved:",
            True,
        )
        print(
            "  Mask remains binary:",
            True,
        )
        print(
            "  Spectral augmentation used:",
            False,
        )

        return 0

    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        AugmentationConfigurationError,
        OSError,
    ) as error:
        LOGGER.error(
            "%s",
            error,
        )
        return 1

    except Exception:
        LOGGER.exception(
            "Unexpected augmentation-audit failure."
        )
        return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )
