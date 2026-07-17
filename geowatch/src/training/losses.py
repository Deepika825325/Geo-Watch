"""Imbalance-aware binary segmentation losses for GeoWatch.

OSCD contains approximately thirty unchanged pixels for every changed pixel.
Plain binary cross-entropy is therefore dominated by easy background pixels.

GeoWatch combines:

* Dice loss, which optimizes change-region overlap.
* Focal loss, which reduces the contribution of easy background pixels.

All losses accept raw model logits. Sigmoid is applied internally only where
needed, preserving the numerical stability of BCE-with-logits.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.nn.functional as functional
import yaml
from torch import Tensor, nn


LOGGER = logging.getLogger(
    "geowatch.losses"
)

VALID_REDUCTIONS = {
    "mean",
    "sum",
    "none",
}


class LossConfigurationError(ValueError):
    """Raised when a loss configuration or tensor contract is invalid."""


class LossBreakdown(NamedTuple):
    """Individual and combined Dice–Focal loss values."""

    total: Tensor
    dice: Tensor
    focal: Tensor


def validate_binary_segmentation_tensors(
    logits: Tensor,
    targets: Tensor,
) -> None:
    """Validate logits and binary segmentation targets.

    Args:
        logits: Raw model output with shape ``[B, 1, H, W]``.
        targets: Binary floating-point mask with the same shape.

    Raises:
        LossConfigurationError: When shapes, dtypes or values are invalid.
    """
    if logits.shape != targets.shape:
        raise LossConfigurationError(
            "Logits and targets must have identical shapes; "
            f"received {tuple(logits.shape)} and {tuple(targets.shape)}."
        )

    if logits.ndim < 2:
        raise LossConfigurationError(
            "Segmentation tensors must contain a batch dimension and "
            "at least one prediction dimension."
        )

    if not logits.is_floating_point():
        raise LossConfigurationError(
            "Logits must be floating-point tensors."
        )

    if not targets.is_floating_point():
        raise LossConfigurationError(
            "Targets must be floating-point binary tensors."
        )

    if not torch.isfinite(
        logits
    ).all():
        raise LossConfigurationError(
            "Logits contain non-finite values."
        )

    if not torch.isfinite(
        targets
    ).all():
        raise LossConfigurationError(
            "Targets contain non-finite values."
        )

    binary_targets = torch.logical_or(
        targets == 0,
        targets == 1,
    )

    if not binary_targets.all():
        raise LossConfigurationError(
            "Targets must contain only binary values 0 and 1."
        )


def reduce_loss(
    values: Tensor,
    reduction: str,
) -> Tensor:
    """Apply a supported reduction to a loss tensor."""
    if reduction == "mean":
        return values.mean()

    if reduction == "sum":
        return values.sum()

    if reduction == "none":
        return values

    raise LossConfigurationError(
        f"Unsupported reduction '{reduction}'. "
        f"Supported reductions are {sorted(VALID_REDUCTIONS)}."
    )


class BinaryDiceLoss(nn.Module):
    """Soft Dice loss calculated from raw binary segmentation logits.

    Dice loss gives each image's foreground overlap direct influence,
    preventing the majority background class from dominating optimization.
    """

    def __init__(
        self,
        smooth: float = 1.0,
        epsilon: float = 1.0e-7,
        include_background: bool = False,
        reduction: str = "mean",
    ) -> None:
        """Initialize binary Dice loss.

        Args:
            smooth: Additive smoothing applied to numerator and denominator.
            epsilon: Numerical protection for the denominator.
            include_background: Average foreground and background Dice when
                true. GeoWatch uses false because change is the target class.
            reduction: ``mean``, ``sum`` or ``none`` over the batch.
        """
        super().__init__()

        if smooth < 0:
            raise LossConfigurationError(
                "Dice smooth must be non-negative."
            )

        if epsilon <= 0:
            raise LossConfigurationError(
                "Dice epsilon must be greater than zero."
            )

        if reduction not in VALID_REDUCTIONS:
            raise LossConfigurationError(
                f"Unsupported Dice reduction: {reduction}"
            )

        self.smooth = float(
            smooth
        )
        self.epsilon = float(
            epsilon
        )
        self.include_background = bool(
            include_background
        )
        self.reduction = reduction

    def _class_dice_loss(
        self,
        probabilities: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """Calculate per-sample Dice loss for one binary class."""
        probabilities_flat = probabilities.flatten(
            start_dim=1
        )
        targets_flat = targets.flatten(
            start_dim=1
        )

        intersection = (
            probabilities_flat
            * targets_flat
        ).sum(
            dim=1
        )

        denominator = (
            probabilities_flat.sum(
                dim=1
            )
            + targets_flat.sum(
                dim=1
            )
        )

        dice_score = (
            2.0 * intersection
            + self.smooth
        ) / (
            denominator
            + self.smooth
            + self.epsilon
        )

        return 1.0 - dice_score

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """Calculate Dice loss from raw logits and binary targets."""
        validate_binary_segmentation_tensors(
            logits=logits,
            targets=targets,
        )

        probabilities = torch.sigmoid(
            logits
        )

        foreground_loss = self._class_dice_loss(
            probabilities=probabilities,
            targets=targets,
        )

        if self.include_background:
            background_loss = self._class_dice_loss(
                probabilities=1.0 - probabilities,
                targets=1.0 - targets,
            )

            per_sample_loss = (
                foreground_loss
                + background_loss
            ) / 2.0
        else:
            per_sample_loss = foreground_loss

        return reduce_loss(
            values=per_sample_loss,
            reduction=self.reduction,
        )


class BinaryFocalLoss(nn.Module):
    """Numerically stable binary focal loss calculated from logits.

    Focal loss multiplies BCE by ``(1 - p_t) ** gamma``. Easy examples with
    high confidence therefore contribute very little, while difficult pixels
    retain a meaningful gradient.
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        """Initialize binary focal loss.

        Args:
            alpha: Weight assigned to changed pixels. Unchanged pixels receive
                ``1 - alpha``.
            gamma: Focusing strength. Zero reduces focal loss to alpha-balanced
                binary cross-entropy.
            reduction: ``mean``, ``sum`` or ``none``.
        """
        super().__init__()

        if not 0.0 <= alpha <= 1.0:
            raise LossConfigurationError(
                "Focal alpha must be between 0 and 1."
            )

        if gamma < 0:
            raise LossConfigurationError(
                "Focal gamma must be non-negative."
            )

        if reduction not in VALID_REDUCTIONS:
            raise LossConfigurationError(
                f"Unsupported Focal reduction: {reduction}"
            )

        self.alpha = float(
            alpha
        )
        self.gamma = float(
            gamma
        )
        self.reduction = reduction

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """Calculate alpha-balanced focal loss from raw logits."""
        validate_binary_segmentation_tensors(
            logits=logits,
            targets=targets,
        )

        binary_cross_entropy = (
            functional.binary_cross_entropy_with_logits(
                logits,
                targets,
                reduction="none",
            )
        )

        probabilities = torch.sigmoid(
            logits
        )

        probability_of_true_class = (
            probabilities * targets
            + (1.0 - probabilities)
            * (1.0 - targets)
        )

        alpha_factor = (
            self.alpha * targets
            + (1.0 - self.alpha)
            * (1.0 - targets)
        )

        modulation = torch.pow(
            1.0 - probability_of_true_class,
            self.gamma,
        )

        focal_values = (
            alpha_factor
            * modulation
            * binary_cross_entropy
        )

        return reduce_loss(
            values=focal_values,
            reduction=self.reduction,
        )


class DiceFocalLoss(nn.Module):
    """Weighted combination of foreground Dice and binary focal loss."""

    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        dice_smooth: float = 1.0,
        dice_epsilon: float = 1.0e-7,
        include_background: bool = False,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        focal_reduction: str = "mean",
    ) -> None:
        """Initialize the combined loss from explicit hyperparameters."""
        super().__init__()

        if dice_weight < 0:
            raise LossConfigurationError(
                "Dice weight must be non-negative."
            )

        if focal_weight < 0:
            raise LossConfigurationError(
                "Focal weight must be non-negative."
            )

        if dice_weight + focal_weight <= 0:
            raise LossConfigurationError(
                "At least one combined-loss weight must be positive."
            )

        if focal_reduction not in {
            "mean",
            "sum",
        }:
            raise LossConfigurationError(
                "Combined Dice–Focal loss requires focal reduction "
                "'mean' or 'sum'."
            )

        self.dice_weight = float(
            dice_weight
        )
        self.focal_weight = float(
            focal_weight
        )

        self.dice_loss = BinaryDiceLoss(
            smooth=dice_smooth,
            epsilon=dice_epsilon,
            include_background=include_background,
            reduction="mean",
        )

        self.focal_loss = BinaryFocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            reduction=focal_reduction,
        )

    def compute_breakdown(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> LossBreakdown:
        """Return combined loss and independently loggable components."""
        dice_value = self.dice_loss(
            logits,
            targets,
        )
        focal_value = self.focal_loss(
            logits,
            targets,
        )

        total = (
            self.dice_weight * dice_value
            + self.focal_weight * focal_value
        )

        return LossBreakdown(
            total=total,
            dice=dice_value,
            focal=focal_value,
        )

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """Return the weighted Dice–Focal objective."""
        return self.compute_breakdown(
            logits=logits,
            targets=targets,
        ).total


def require_mapping(
    value: Any,
    name: str,
) -> Mapping[str, Any]:
    """Validate and return one configuration mapping."""
    if not isinstance(
        value,
        Mapping,
    ):
        raise LossConfigurationError(
            f"{name} must be a configuration mapping."
        )

    return value


def build_loss_from_config(
    config: Mapping[str, Any],
) -> DiceFocalLoss:
    """Build Dice–Focal loss from the root or loss-only YAML mapping."""
    loss_config_value = config.get(
        "loss",
        config,
    )
    loss_config = require_mapping(
        loss_config_value,
        "loss",
    )

    loss_name = str(
        loss_config.get(
            "name",
            "",
        )
    ).strip().lower()

    if loss_name != "dice_focal":
        raise LossConfigurationError(
            "GeoWatch Week 4 requires loss.name='dice_focal'; "
            f"received '{loss_name}'."
        )

    dice_config = require_mapping(
        loss_config.get(
            "dice"
        ),
        "loss.dice",
    )
    focal_config = require_mapping(
        loss_config.get(
            "focal"
        ),
        "loss.focal",
    )

    return DiceFocalLoss(
        dice_weight=float(
            dice_config[
                "weight"
            ]
        ),
        focal_weight=float(
            focal_config[
                "weight"
            ]
        ),
        dice_smooth=float(
            dice_config[
                "smooth"
            ]
        ),
        dice_epsilon=float(
            dice_config[
                "epsilon"
            ]
        ),
        include_background=bool(
            dice_config[
                "include_background"
            ]
        ),
        focal_alpha=float(
            focal_config[
                "alpha"
            ]
        ),
        focal_gamma=float(
            focal_config[
                "gamma"
            ]
        ),
        focal_reduction=str(
            focal_config[
                "reduction"
            ]
        ),
    )


def load_yaml_config(
    path: Path,
) -> Mapping[str, Any]:
    """Load a UTF-8 YAML configuration document."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Training configuration not found: {path}"
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
    """Build the loss-audit command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Audit GeoWatch Dice and Focal losses using deterministic "
            "synthetic binary segmentation tensors."
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
    """Run a deterministic configuration and gradient audit."""
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
        criterion = build_loss_from_config(
            config
        )

        generator = torch.Generator().manual_seed(
            args.seed
        )

        logits = torch.randn(
            2,
            1,
            32,
            32,
            generator=generator,
            requires_grad=True,
        )

        targets = torch.zeros_like(
            logits
        )
        targets[
            :,
            :,
            8:16,
            8:16,
        ] = 1.0

        breakdown = criterion.compute_breakdown(
            logits=logits,
            targets=targets,
        )

        breakdown.total.backward()

        if logits.grad is None:
            raise LossConfigurationError(
                "No gradient reached the synthetic logits."
            )

        if not torch.isfinite(
            logits.grad
        ).all():
            raise LossConfigurationError(
                "Non-finite loss gradient detected."
            )

        print(
            "GeoWatch Dice–Focal loss audit passed"
        )
        print(
            "  Configuration:",
            args.config,
        )
        print(
            "  Dice weight:",
            criterion.dice_weight,
        )
        print(
            "  Focal weight:",
            criterion.focal_weight,
        )
        print(
            "  Focal alpha:",
            criterion.focal_loss.alpha,
        )
        print(
            "  Focal gamma:",
            criterion.focal_loss.gamma,
        )
        print(
            "  Target change fraction:",
            float(
                targets.mean().item()
            ),
        )
        print(
            "  Dice loss:",
            float(
                breakdown.dice.item()
            ),
        )
        print(
            "  Focal loss:",
            float(
                breakdown.focal.item()
            ),
        )
        print(
            "  Combined loss:",
            float(
                breakdown.total.item()
            ),
        )
        print(
            "  Gradient finite:",
            True,
        )

        return 0

    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        LossConfigurationError,
        OSError,
    ) as error:
        LOGGER.error(
            "%s",
            error,
        )
        return 1

    except Exception:
        LOGGER.exception(
            "Unexpected loss-audit failure."
        )
        return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )
