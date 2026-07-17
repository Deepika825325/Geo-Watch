"""Tests for GeoWatch imbalance-aware segmentation losses."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as functional
import yaml

from src.training.losses import (
    BinaryBCELoss,
    BinaryDiceLoss,
    BinaryFocalLoss,
    DiceFocalLoss,
    LossConfigurationError,
    build_loss_from_config,
)


def create_binary_targets() -> torch.Tensor:
    """Create a deterministic foreground/background mask."""
    return torch.tensor(
        [
            [
                [
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                ],
            ]
        ],
        dtype=torch.float32,
    )


def logits_matching_targets(
    targets: torch.Tensor,
    magnitude: float = 20.0,
) -> torch.Tensor:
    """Create highly confident logits matching a binary target."""
    return torch.where(
        targets == 1,
        torch.full_like(
            targets,
            magnitude,
        ),
        torch.full_like(
            targets,
            -magnitude,
        ),
    )


def test_perfect_predictions_have_near_zero_loss() -> None:
    """Both components should approach zero for correct confident logits."""
    targets = create_binary_targets()
    logits = logits_matching_targets(
        targets
    )

    dice = BinaryDiceLoss()(
        logits,
        targets,
    )
    focal = BinaryFocalLoss()(
        logits,
        targets,
    )

    assert float(
        dice.item()
    ) < 1.0e-6
    assert float(
        focal.item()
    ) < 1.0e-8


def test_incorrect_predictions_have_higher_combined_loss() -> None:
    """Inverting the prediction must substantially increase the objective."""
    targets = create_binary_targets()

    correct_logits = logits_matching_targets(
        targets,
        magnitude=8.0,
    )
    incorrect_logits = -correct_logits

    criterion = DiceFocalLoss()

    correct_loss = criterion(
        correct_logits,
        targets,
    )
    incorrect_loss = criterion(
        incorrect_logits,
        targets,
    )

    assert incorrect_loss > correct_loss
    assert float(
        incorrect_loss.item()
    ) > 1.0


def test_empty_target_and_empty_prediction_are_stable() -> None:
    """No-change patches must not create NaN or an artificial Dice penalty."""
    targets = torch.zeros(
        2,
        1,
        8,
        8,
    )
    logits = torch.full_like(
        targets,
        -20.0,
    )

    loss = BinaryDiceLoss()(
        logits,
        targets,
    )

    assert torch.isfinite(
        loss
    )
    assert float(
        loss.item()
    ) < 1.0e-6


def test_focal_loss_downweights_easy_background() -> None:
    """Easy unchanged pixels must contribute less than ordinary BCE."""
    logits = torch.full(
        (
            1,
            1,
            8,
            8,
        ),
        -5.0,
    )
    targets = torch.zeros_like(
        logits
    )

    focal = BinaryFocalLoss(
        alpha=0.75,
        gamma=2.0,
    )(
        logits,
        targets,
    )

    bce = functional.binary_cross_entropy_with_logits(
        logits,
        targets,
    )

    assert focal < bce


def test_alpha_emphasizes_changed_pixels() -> None:
    """Equivalent hard positives receive more weight than hard negatives."""
    focal = BinaryFocalLoss(
        alpha=0.75,
        gamma=2.0,
    )

    hard_positive = focal(
        torch.tensor(
            [[[[-1.0]]]]
        ),
        torch.tensor(
            [[[[1.0]]]]
        ),
    )

    hard_negative = focal(
        torch.tensor(
            [[[[1.0]]]]
        ),
        torch.tensor(
            [[[[0.0]]]]
        ),
    )

    ratio = float(
        (
            hard_positive
            / hard_negative
        ).item()
    )

    assert ratio == pytest.approx(
        3.0,
        rel=1.0e-5,
    )


def test_combined_loss_produces_finite_gradients() -> None:
    """Dice and Focal must jointly backpropagate through raw logits."""
    generator = torch.Generator().manual_seed(
        42
    )

    logits = torch.randn(
        2,
        1,
        16,
        16,
        generator=generator,
        requires_grad=True,
    )
    targets = torch.zeros_like(
        logits
    )
    targets[
        :,
        :,
        4:8,
        4:8,
    ] = 1.0

    criterion = DiceFocalLoss()
    breakdown = criterion.compute_breakdown(
        logits,
        targets,
    )

    breakdown.total.backward()

    assert logits.grad is not None
    assert torch.isfinite(
        logits.grad
    ).all()
    assert torch.isfinite(
        breakdown.total
    )
    assert torch.isfinite(
        breakdown.dice
    )
    assert torch.isfinite(
        breakdown.focal
    )


def test_loss_is_built_from_frozen_yaml_config() -> None:
    """The runtime loss must inherit every value from train_config.yaml."""
    config_path = Path(
        "configs/train_config.yaml"
    )

    with config_path.open(
        "r",
        encoding="utf-8-sig",
    ) as config_file:
        config = yaml.safe_load(
            config_file
        )

    criterion = build_loss_from_config(
        config
    )

    assert criterion.dice_weight == pytest.approx(
        0.5
    )
    assert criterion.focal_weight == pytest.approx(
        0.5
    )
    assert criterion.dice_loss.smooth == pytest.approx(
        1.0
    )
    assert criterion.dice_loss.epsilon == pytest.approx(
        1.0e-7
    )
    assert criterion.dice_loss.include_background is False
    assert criterion.focal_loss.alpha == pytest.approx(
        0.75
    )
    assert criterion.focal_loss.gamma == pytest.approx(
        2.0
    )
    assert criterion.focal_loss.reduction == "mean"


def test_shape_mismatch_is_rejected() -> None:
    """Loss evaluation must reject spatially misaligned masks."""
    logits = torch.zeros(
        1,
        1,
        8,
        8,
    )
    targets = torch.zeros(
        1,
        1,
        8,
        7,
    )

    with pytest.raises(
        LossConfigurationError,
        match="identical shapes",
    ):
        BinaryDiceLoss()(
            logits,
            targets,
        )


def test_nonbinary_targets_are_rejected() -> None:
    """Interpolated or corrupted targets must not enter the loss silently."""
    logits = torch.zeros(
        1,
        1,
        4,
        4,
    )
    targets = torch.zeros_like(
        logits
    )
    targets[
        0,
        0,
        0,
        0,
    ] = 0.5

    with pytest.raises(
        LossConfigurationError,
        match="only binary values",
    ):
        BinaryFocalLoss()(
            logits,
            targets,
        )


def test_invalid_hyperparameters_are_rejected() -> None:
    """Unsafe Dice, Focal and combined settings must fail immediately."""
    with pytest.raises(
        LossConfigurationError,
        match="between 0 and 1",
    ):
        BinaryFocalLoss(
            alpha=1.5
        )

    with pytest.raises(
        LossConfigurationError,
        match="non-negative",
    ):
        BinaryFocalLoss(
            gamma=-1.0
        )

    with pytest.raises(
        LossConfigurationError,
        match="greater than zero",
    ):
        BinaryDiceLoss(
            epsilon=0.0
        )

    with pytest.raises(
        LossConfigurationError,
        match="At least one",
    ):
        DiceFocalLoss(
            dice_weight=0.0,
            focal_weight=0.0,
        )

def test_plain_bce_matches_pytorch_reference() -> None:
    """GeoWatch plain BCE must equal PyTorch BCE-with-logits exactly."""

    logits = torch.tensor(
        [
            [
                [
                    [
                        -1.0,
                        0.5,
                        2.0,
                    ]
                ]
            ]
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor(
        [
            [
                [
                    [
                        0.0,
                        1.0,
                        1.0,
                    ]
                ]
            ]
        ],
        dtype=torch.float32,
    )

    criterion = BinaryBCELoss(
        reduction="mean"
    )

    breakdown = criterion.compute_breakdown(
        logits=logits,
        targets=targets,
    )

    expected = functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="mean",
    )

    torch.testing.assert_close(
        breakdown.total,
        expected,
        rtol=1.0e-7,
        atol=1.0e-7,
    )
    assert float(
        breakdown.dice.item()
    ) == pytest.approx(
        0.0
    )
    assert float(
        breakdown.focal.item()
    ) == pytest.approx(
        0.0
    )

    breakdown.total.backward()

    assert logits.grad is not None
    assert torch.isfinite(
        logits.grad
    ).all()


def test_plain_bce_is_built_from_config() -> None:
    """The loss factory must construct exact unweighted BCE."""

    criterion = build_loss_from_config(
        {
            "loss": {
                "name": "bce",
                "reduction": "mean",
            }
        }
    )

    assert isinstance(
        criterion,
        BinaryBCELoss,
    )
    assert criterion.reduction == "mean"
