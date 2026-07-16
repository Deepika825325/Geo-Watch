"""Architecture tests for the GeoWatch Siamese U-Net.

All tests use deterministic synthetic tensors. OSCD training images,
testing images and labels are intentionally excluded from architecture
debugging.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch
from torch import Tensor

from src.models.encoder import (
    EncoderFeatures,
    MultispectralResNet18Encoder,
)
from src.models.siamese_unet import (
    ModelError,
    SiameseUNet,
    fuse_absolute_difference,
)


TEST_DECODER_CHANNELS = (
    64,
    32,
    16,
    16,
)
TEST_HEAD_CHANNELS = 8
TEST_IMAGE_SIZE = 64


def build_test_model(
    input_channels: int = 4,
    band_names: Sequence[str] | None = None,
) -> SiameseUNet:
    """Build a compact deterministic model for CPU unit tests.

    The ResNet-18 encoder remains unchanged. Only decoder width is reduced
    so architecture tests run faster on CPU.
    """
    model = SiameseUNet(
        input_channels=input_channels,
        band_names=band_names,
        pretrained_encoder=False,
        decoder_channels=TEST_DECODER_CHANNELS,
        head_channels=TEST_HEAD_CHANNELS,
        preferred_norm_groups=8,
        dropout_probability=0.0,
    )

    model.eval()

    return model


def make_pair(
    input_channels: int,
    batch_size: int = 1,
    height: int = TEST_IMAGE_SIZE,
    width: int = TEST_IMAGE_SIZE,
    requires_grad: bool = False,
) -> tuple[Tensor, Tensor]:
    """Create a deterministic synthetic bi-temporal tensor pair."""
    generator = torch.Generator().manual_seed(
        42
    )

    before = torch.randn(
        batch_size,
        input_channels,
        height,
        width,
        generator=generator,
    )
    after = torch.randn(
        batch_size,
        input_channels,
        height,
        width,
        generator=generator,
    )

    if requires_grad:
        before.requires_grad_()
        after.requires_grad_()

    return before, after


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
def test_output_shape_for_supported_channel_counts(
    input_channels: int,
    expected_bands: tuple[str, ...],
) -> None:
    """The model must restore one full-resolution logit per pixel."""
    model = build_test_model(
        input_channels=input_channels
    )
    before, after = make_pair(
        input_channels=input_channels
    )

    with torch.inference_mode():
        logits = model(
            before,
            after,
        )

    assert tuple(logits.shape) == (
        1,
        1,
        TEST_IMAGE_SIZE,
        TEST_IMAGE_SIZE,
    )
    assert logits.dtype == before.dtype
    assert model.band_names == expected_bands


def test_model_uses_one_shared_encoder() -> None:
    """Only one encoder parameter namespace may exist."""
    model = build_test_model()

    module_names = tuple(
        name
        for name, _ in model.named_modules()
    )
    state_keys = tuple(
        model.state_dict().keys()
    )

    assert isinstance(
        model.encoder,
        MultispectralResNet18Encoder,
    )
    assert "encoder" in module_names

    assert any(
        key.startswith("encoder.")
        for key in state_keys
    )
    assert not any(
        key.startswith("encoder_before.")
        for key in state_keys
    )
    assert not any(
        key.startswith("encoder_after.")
        for key in state_keys
    )

    assert not hasattr(
        model,
        "encoder_before",
    )
    assert not hasattr(
        model,
        "encoder_after",
    )


def test_absolute_difference_fusion() -> None:
    """Every fused scale must equal the exact absolute feature difference."""
    before_features = EncoderFeatures(
        stem=torch.tensor(
            [[[[1.0, 4.0]]]]
        ),
        stage1=torch.tensor(
            [[[[2.0]]]]
        ),
        stage2=torch.tensor(
            [[[[3.0]]]]
        ),
        stage3=torch.tensor(
            [[[[4.0]]]]
        ),
        stage4=torch.tensor(
            [[[[5.0]]]]
        ),
    )

    after_features = EncoderFeatures(
        stem=torch.tensor(
            [[[[3.0, 1.0]]]]
        ),
        stage1=torch.tensor(
            [[[[5.0]]]]
        ),
        stage2=torch.tensor(
            [[[[1.0]]]]
        ),
        stage3=torch.tensor(
            [[[[9.0]]]]
        ),
        stage4=torch.tensor(
            [[[[2.0]]]]
        ),
    )

    fused = fuse_absolute_difference(
        before_features=before_features,
        after_features=after_features,
    )

    expected = (
        torch.tensor(
            [[[[2.0, 3.0]]]]
        ),
        torch.tensor(
            [[[[3.0]]]]
        ),
        torch.tensor(
            [[[[2.0]]]]
        ),
        torch.tensor(
            [[[[5.0]]]]
        ),
        torch.tensor(
            [[[[3.0]]]]
        ),
    )

    for actual_feature, expected_feature in zip(
        fused,
        expected,
        strict=True,
    ):
        assert torch.equal(
            actual_feature,
            expected_feature,
        )


def test_date_order_symmetry() -> None:
    """Absolute-difference fusion must make the model order-symmetric."""
    torch.manual_seed(
        42
    )

    model = build_test_model()
    before, after = make_pair(
        input_channels=4
    )

    with torch.inference_mode():
        forward_logits = model(
            before,
            after,
        )
        swapped_logits = model(
            after,
            before,
        )

    assert torch.allclose(
        forward_logits,
        swapped_logits,
        rtol=1e-5,
        atol=1e-6,
    )

    maximum_error = torch.max(
        torch.abs(
            forward_logits
            - swapped_logits
        )
    )

    assert float(
        maximum_error.item()
    ) <= 1e-6


def test_identical_inputs_produce_finite_logits() -> None:
    """No-change synthetic pairs must remain numerically stable."""
    model = build_test_model()

    image, _ = make_pair(
        input_channels=4
    )

    with torch.inference_mode():
        logits = model(
            image,
            image.clone(),
        )

    assert tuple(logits.shape) == (
        1,
        1,
        TEST_IMAGE_SIZE,
        TEST_IMAGE_SIZE,
    )
    assert torch.isfinite(
        logits
    ).all()


def test_gradients_reach_both_inputs_encoder_and_head() -> None:
    """The full architecture must be differentiable end to end."""
    torch.manual_seed(
        42
    )

    model = build_test_model()
    before, after = make_pair(
        input_channels=4,
        requires_grad=True,
    )

    logits = model(
        before,
        after,
    )

    loss = logits.square().mean()
    loss.backward()

    assert before.grad is not None
    assert after.grad is not None

    assert torch.isfinite(
        before.grad
    ).all()
    assert torch.isfinite(
        after.grad
    ).all()

    encoder_weight = (
        model.encoder.conv1.weight
    )
    classifier_weight = (
        model.segmentation_head.classifier.weight
    )

    assert encoder_weight.grad is not None
    assert classifier_weight.grad is not None

    assert torch.isfinite(
        encoder_weight.grad
    ).all()
    assert torch.isfinite(
        classifier_weight.grad
    ).all()


@pytest.mark.parametrize(
    (
        "before_shape",
        "after_shape",
        "expected_message",
    ),
    (
        (
            (
                1,
                4,
                64,
                64,
            ),
            (
                1,
                4,
                64,
                96,
            ),
            "identical shapes",
        ),
        (
            (
                1,
                6,
                64,
                64,
            ),
            (
                1,
                6,
                64,
                64,
            ),
            "expects 4 channels",
        ),
        (
            (
                1,
                4,
                64,
                80,
            ),
            (
                1,
                4,
                64,
                80,
            ),
            "divisible by 32",
        ),
    ),
)
def test_invalid_input_contract_is_rejected(
    before_shape: tuple[int, ...],
    after_shape: tuple[int, ...],
    expected_message: str,
) -> None:
    """Malformed temporal pairs must fail before encoder execution."""
    model = build_test_model()

    before = torch.randn(
        *before_shape
    )
    after = torch.randn(
        *after_shape
    )

    with pytest.raises(
        ModelError,
        match=expected_message,
    ):
        model(
            before,
            after,
        )


def test_integer_inputs_are_rejected() -> None:
    """Raw integer satellite arrays must be normalized before the model."""
    model = build_test_model()

    before = torch.zeros(
        1,
        4,
        TEST_IMAGE_SIZE,
        TEST_IMAGE_SIZE,
        dtype=torch.int16,
    )
    after = torch.zeros_like(
        before
    )

    with pytest.raises(
        ModelError,
        match="floating-point",
    ):
        model(
            before,
            after,
        )


def test_model_returns_logits_without_sigmoid_layer() -> None:
    """The architecture must leave sigmoid application to loss/inference."""
    model = build_test_model()

    sigmoid_modules = [
        module
        for module in model.modules()
        if isinstance(
            module,
            torch.nn.Sigmoid,
        )
    ]

    assert sigmoid_modules == []
    assert (
        model.segmentation_head.classifier.out_channels
        == 1
    )
