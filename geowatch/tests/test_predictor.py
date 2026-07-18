from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from affine import Affine
from rasterio.crs import CRS
from torch import Tensor, nn

from src.inference.predictor import (
    FROZEN_BANDS,
    FROZEN_ONNX_SHA256,
    FROZEN_THRESHOLD,
    FrozenChangePredictor,
    InferenceBackend,
    RasterPairError,
    calculate_starts,
    read_aligned_pair,
    resolve_backend,
    run_tiled_inference,
    run_tiled_onnx_inference,
)


class DifferenceModel(nn.Module):
    def forward(
        self,
        before: Tensor,
        after: Tensor,
    ) -> Tensor:
        return (
            after[
                :,
                0:1,
            ]
            - before[
                :,
                0:1,
            ]
        ) * 10.0


def write_single_band(
    path: Path,
    value: int,
    height: int,
    width: int,
    transform: Affine,
    crs: CRS,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    array = np.full(
        (
            height,
            width,
        ),
        value,
        dtype=np.uint16,
    )

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint16",
        crs=crs,
        transform=transform,
    ) as dataset:
        dataset.write(
            array,
            1,
        )


def build_pair(
    root: Path,
    before_value: int = 0,
    after_value: int = 10_000,
    height: int = 300,
    width: int = 500,
) -> tuple[Path, Path]:
    transform = Affine.translation(
        78.0,
        18.0,
    ) * Affine.scale(
        10.0,
        -10.0,
    )

    crs = CRS.from_epsg(
        32644
    )

    before = (
        root
        / "before"
    )

    after = (
        root
        / "after"
    )

    for band in FROZEN_BANDS:
        write_single_band(
            before
            / f"{band}.tif",
            value=before_value,
            height=height,
            width=width,
            transform=transform,
            crs=crs,
        )

        write_single_band(
            after
            / f"{band}.tif",
            value=after_value,
            height=height,
            width=width,
            transform=transform,
            crs=crs,
        )

    return (
        before,
        after,
    )


def test_calculate_starts_exact_grid(
) -> None:
    assert calculate_starts(
        500,
        256,
        256,
    ) == (
        0,
        256,
    )

    assert calculate_starts(
        512,
        256,
        256,
    ) == (
        0,
        256,
    )

    with pytest.raises(
        ValueError,
        match="stride equal to patch size",
    ):
        calculate_starts(
            500,
            256,
            128,
        )


def test_read_aligned_pair_normalizes_reflectance(
    tmp_path: Path,
) -> None:
    before, after = build_pair(
        tmp_path
    )

    pair = read_aligned_pair(
        before,
        after,
    )

    assert pair.before.shape == (
        4,
        300,
        500,
    )

    assert pair.after.shape == (
        4,
        300,
        500,
    )

    assert float(
        pair.before.min()
    ) == pytest.approx(
        0.0
    )

    assert float(
        pair.after.max()
    ) == pytest.approx(
        1.0
    )

    assert pair.metadata.height == 300
    assert pair.metadata.width == 500
    assert pair.metadata.crs == CRS.from_epsg(
        32644
    )


def test_read_aligned_pair_rejects_transform_mismatch(
    tmp_path: Path,
) -> None:
    before, after = build_pair(
        tmp_path
    )

    mismatched_transform = Affine.translation(
        79.0,
        18.0,
    ) * Affine.scale(
        10.0,
        -10.0,
    )

    write_single_band(
        after
        / "B08.tif",
        value=10_000,
        height=300,
        width=500,
        transform=mismatched_transform,
        crs=CRS.from_epsg(
            32644
        ),
    )

    with pytest.raises(
        RasterPairError,
        match="transform mismatch",
    ):
        read_aligned_pair(
            before,
            after,
        )


def test_tiled_inference_restores_original_shape(
) -> None:
    before = np.zeros(
        (
            4,
            300,
            500,
        ),
        dtype=np.float32,
    )

    after = np.ones(
        (
            4,
            300,
            500,
        ),
        dtype=np.float32,
    )

    probability, mask, patch_count = run_tiled_inference(
        model=DifferenceModel(),
        before=before,
        after=after,
        device=torch.device(
            "cpu"
        ),
        batch_size=3,
        threshold=FROZEN_THRESHOLD,
    )

    assert probability.shape == (
        300,
        500,
    )

    assert mask.shape == (
        300,
        500,
    )

    assert patch_count == 4

    assert np.isfinite(
        probability
    ).all()

    assert float(
        probability.min()
    ) > FROZEN_THRESHOLD

    assert np.all(
        mask
        == 1
    )


def test_tiled_inference_rejects_threshold_change(
) -> None:
    before = np.zeros(
        (
            4,
            256,
            256,
        ),
        dtype=np.float32,
    )

    after = np.ones(
        (
            4,
            256,
            256,
        ),
        dtype=np.float32,
    )

    with pytest.raises(
        RuntimeError,
        match="threshold must remain frozen",
    ):
        run_tiled_inference(
            model=DifferenceModel(),
            before=before,
            after=after,
            device=torch.device(
                "cpu"
            ),
            batch_size=1,
            threshold=0.50,
        )

def test_resolve_backend_honors_configured_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "MODEL_BACKEND",
        InferenceBackend.ONNX_CPU,
    )

    assert resolve_backend() == InferenceBackend.ONNX_CPU

    monkeypatch.setenv(
        "MODEL_BACKEND",
        InferenceBackend.CUDA,
    )

    assert resolve_backend() == InferenceBackend.CUDA


def test_resolve_backend_explicit_value_overrides_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "MODEL_BACKEND",
        InferenceBackend.CUDA,
    )

    assert resolve_backend(
        InferenceBackend.ONNX_CPU
    ) == InferenceBackend.ONNX_CPU


def test_resolve_backend_rejects_unknown_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "MODEL_BACKEND",
        "invalid",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported model backend",
    ):
        resolve_backend()


class DifferenceOnnxSession:
    def run(
        self,
        output_names: list[str],
        inputs: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        if output_names != [
            "logits",
        ]:
            raise RuntimeError(
                "Unexpected output names."
            )

        before = inputs[
            "before"
        ]

        after = inputs[
            "after"
        ]

        logits = (
            after[
                :,
                0:1,
            ]
            - before[
                :,
                0:1,
            ]
        ) * np.float32(
            10.0
        )

        return [
            logits.astype(
                np.float32,
                copy=False,
            )
        ]


def test_onnx_tiled_inference_matches_torch_wrapper(
) -> None:
    before = np.zeros(
        (
            4,
            300,
            500,
        ),
        dtype=np.float32,
    )

    after = np.ones(
        (
            4,
            300,
            500,
        ),
        dtype=np.float32,
    )

    (
        torch_probability,
        torch_mask,
        torch_patch_count,
    ) = run_tiled_inference(
        model=DifferenceModel(),
        before=before,
        after=after,
        device=torch.device(
            "cpu"
        ),
        batch_size=3,
    )

    (
        onnx_probability,
        onnx_mask,
        onnx_patch_count,
    ) = run_tiled_onnx_inference(
        session=DifferenceOnnxSession(),
        before=before,
        after=after,
        batch_size=3,
    )

    np.testing.assert_allclose(
        onnx_probability,
        torch_probability,
        rtol=1e-6,
        atol=1e-7,
    )

    np.testing.assert_array_equal(
        onnx_mask,
        torch_mask,
    )

    assert onnx_patch_count == torch_patch_count


def test_frozen_predictor_runs_real_onnx_backend(
    tmp_path: Path,
) -> None:
    before, after = build_pair(
        tmp_path,
        before_value=1_000,
        after_value=9_000,
        height=256,
        width=256,
    )

    predictor = FrozenChangePredictor(
        onnx_model_path=Path(
            "deploy/model.onnx"
        ),
        backend=InferenceBackend.ONNX_CPU,
        device="cpu",
        batch_size=1,
    )

    result = predictor.predict_pair(
        before_directory=before,
        after_directory=after,
        qualitative=True,
    )

    assert predictor.backend == InferenceBackend.ONNX_CPU
    assert predictor.device == "cpu"
    assert predictor.onnx_model_sha256 == FROZEN_ONNX_SHA256
    assert result.probability.shape == (
        256,
        256,
    )
    assert result.mask.shape == (
        256,
        256,
    )
    assert result.patch_count == 1
    assert result.threshold == FROZEN_THRESHOLD
    assert result.qualitative is True
    assert np.isfinite(
        result.probability
    ).all()
