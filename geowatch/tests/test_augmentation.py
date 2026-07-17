"""Tests for GeoWatch paired geometric augmentation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from src.training.augmentation import (
    AugmentationConfigurationError,
    GeometricDecision,
    PairedGeometricAugmentation,
    apply_geometric_decision,
    apply_paired_geometric_decision,
    build_augmentation_from_config,
)


def create_sample() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Create an aligned synthetic multispectral pair and binary mask."""
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
        1:4,
        2:6,
    ] = 1.0

    return (
        before,
        after,
        mask,
    )


def test_disabled_augmentation_is_a_noop() -> None:
    """Disabling augmentation must preserve every tensor exactly."""
    before, after, mask = create_sample()

    augmentation = PairedGeometricAugmentation(
        enabled=False,
    )

    transformed = augmentation(
        before,
        after,
        mask,
    )

    assert torch.equal(
        transformed[0],
        before,
    )
    assert torch.equal(
        transformed[1],
        after,
    )
    assert torch.equal(
        transformed[2],
        mask,
    )


@pytest.mark.parametrize(
    "decision",
    (
        GeometricDecision(
            horizontal_flip=True,
            vertical_flip=False,
            quarter_turns=0,
        ),
        GeometricDecision(
            horizontal_flip=False,
            vertical_flip=True,
            quarter_turns=0,
        ),
        GeometricDecision(
            horizontal_flip=True,
            vertical_flip=True,
            quarter_turns=1,
        ),
    ),
)
def test_explicit_decision_is_shared_across_all_tensors(
    decision: GeometricDecision,
) -> None:
    """Before, after and mask must receive the identical transformation."""
    before, after, mask = create_sample()

    transformed_before, transformed_after, transformed_mask = (
        apply_paired_geometric_decision(
            before=before,
            after=after,
            mask=mask,
            decision=decision,
        )
    )

    assert torch.equal(
        transformed_before,
        apply_geometric_decision(
            before,
            decision,
        ),
    )
    assert torch.equal(
        transformed_after,
        apply_geometric_decision(
            after,
            decision,
        ),
    )
    assert torch.equal(
        transformed_mask,
        apply_geometric_decision(
            mask,
            decision,
        ),
    )

    assert torch.all(
        transformed_after
        - transformed_before
        == 1_000.0
    )


def test_sampling_is_reproducible_with_identical_seed() -> None:
    """Explicit PyTorch generators must reproduce augmentation decisions."""
    augmentation = PairedGeometricAugmentation(
        horizontal_flip_probability=0.5,
        vertical_flip_probability=0.5,
        rotate_90_probability=0.5,
    )

    first_generator = torch.Generator().manual_seed(
        42
    )
    second_generator = torch.Generator().manual_seed(
        42
    )

    first_decision = augmentation.sample_decision(
        generator=first_generator
    )
    second_decision = augmentation.sample_decision(
        generator=second_generator
    )

    assert first_decision == second_decision


def test_reflectance_values_and_binary_mask_are_preserved() -> None:
    """Exact geometric operations must not alter pixel values."""
    before, after, mask = create_sample()

    decision = GeometricDecision(
        horizontal_flip=True,
        vertical_flip=True,
        quarter_turns=3,
    )

    transformed_before, transformed_after, transformed_mask = (
        apply_paired_geometric_decision(
            before,
            after,
            mask,
            decision,
        )
    )

    assert torch.equal(
        torch.sort(
            before.flatten()
        ).values,
        torch.sort(
            transformed_before.flatten()
        ).values,
    )
    assert torch.equal(
        torch.sort(
            after.flatten()
        ).values,
        torch.sort(
            transformed_after.flatten()
        ).values,
    )
    assert set(
        torch.unique(
            transformed_mask
        ).tolist()
    ) == {
        0.0,
        1.0,
    }


def test_augmentation_preserves_image_gradients() -> None:
    """Flip and rotation operations must remain differentiable."""
    before, after, mask = create_sample()

    before.requires_grad_()
    after.requires_grad_()

    decision = GeometricDecision(
        horizontal_flip=True,
        vertical_flip=False,
        quarter_turns=2,
    )

    transformed_before, transformed_after, _ = (
        apply_paired_geometric_decision(
            before,
            after,
            mask,
            decision,
        )
    )

    loss = (
        transformed_before.mean()
        + transformed_after.mean()
    )
    loss.backward()

    assert before.grad is not None
    assert after.grad is not None
    assert torch.isfinite(
        before.grad
    ).all()
    assert torch.isfinite(
        after.grad
    ).all()


def test_augmentation_is_built_from_frozen_yaml() -> None:
    """Runtime probabilities must come from train_config.yaml."""
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

    augmentation = build_augmentation_from_config(
        config
    )

    assert augmentation.enabled is True
    assert (
        augmentation.horizontal_flip_probability
        == pytest.approx(
            0.5
        )
    )
    assert (
        augmentation.vertical_flip_probability
        == pytest.approx(
            0.5
        )
    )
    assert (
        augmentation.rotate_90_probability
        == pytest.approx(
            0.5
        )
    )


def test_reflectance_changing_configuration_is_rejected() -> None:
    """Colour or spectral augmentation must fail configuration validation."""
    config = {
        "augmentation": {
            "enabled": True,
            "horizontal_flip_probability": 0.5,
            "vertical_flip_probability": 0.5,
            "rotate_90_probability": 0.5,
            "color_jitter_enabled": True,
        }
    }

    with pytest.raises(
        AugmentationConfigurationError,
        match="forbids reflectance-changing",
    ):
        build_augmentation_from_config(
            config
        )


def test_mismatched_temporal_shapes_are_rejected() -> None:
    """Spatially inconsistent temporal images must never be transformed."""
    before = torch.zeros(
        4,
        8,
        8,
    )
    after = torch.zeros(
        4,
        8,
        7,
    )
    mask = torch.zeros(
        1,
        8,
        8,
    )

    with pytest.raises(
        AugmentationConfigurationError,
        match="identical shapes",
    ):
        apply_paired_geometric_decision(
            before,
            after,
            mask,
            GeometricDecision(
                False,
                False,
                0,
            ),
        )


def test_odd_rotation_rejects_nonsquare_patch() -> None:
    """A 90-degree turn must not silently change configured patch shape."""
    before = torch.zeros(
        4,
        8,
        12,
    )
    after = torch.zeros_like(
        before
    )
    mask = torch.zeros(
        1,
        8,
        12,
    )

    with pytest.raises(
        AugmentationConfigurationError,
        match="require square patches",
    ):
        apply_paired_geometric_decision(
            before,
            after,
            mask,
            GeometricDecision(
                False,
                False,
                1,
            ),
        )
