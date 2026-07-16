"""Final reproducible audit for GeoWatch Week 3.

The audit compiles the source tree, runs the automated tests, validates
dependencies, checks Git whitespace, loads one official OSCD training patch,
and passes that patch through the weight-shared Siamese U-Net.

No OSCD test-label path is requested or accessed.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

from src.data.oscd_dataset import OSCDTrainingDataset
from src.models.siamese_unet import SiameseUNet, count_parameters


LOGGER = logging.getLogger(
    "geowatch.week3_audit"
)


class Week3AuditError(RuntimeError):
    """Raised when a Week 3 completion check fails."""


def run_command(
    command: list[str],
    repository_root: Path,
) -> str:
    """Run a required audit command and return its output.

    Args:
        command: Command and arguments to execute.
        repository_root: GeoWatch repository directory.

    Returns:
        Combined standard output and standard error.

    Raises:
        Week3AuditError: When the command exits unsuccessfully.
    """
    result = subprocess.run(
        command,
        cwd=repository_root,
        text=True,
        capture_output=True,
        check=False,
    )

    output = "\n".join(
        part.strip()
        for part in (
            result.stdout,
            result.stderr,
        )
        if part.strip()
    )

    if result.returncode != 0:
        raise Week3AuditError(
            "Audit command failed:\n"
            f"  {' '.join(command)}\n\n"
            f"{output}"
        )

    return output


def final_nonempty_line(
    text: str,
) -> str:
    """Return the final non-empty line from command output."""
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    return (
        lines[-1]
        if lines
        else ""
    )


def write_json_atomic(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write a JSON document atomically."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(
        path
    )


def write_text_atomic(
    path: Path,
    content: str,
) -> None:
    """Write a UTF-8 text document atomically."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        content.rstrip() + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(
        path
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the Week 3 audit command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the final GeoWatch Week 3 architecture and "
            "training-dataset audit."
        )
    )

    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(
            "data/benchmark/oscd/raw"
        ),
    )
    parser.add_argument(
        "--region",
        default="abudhabi",
        help=(
            "Official OSCD training region used for the "
            "real-data integration check."
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(
            "reports/week3"
        ),
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
    """Execute the complete Week 3 audit."""
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
        repository_root = (
            args.repository_root
            .resolve()
        )

        raw_root = args.raw_root

        if not raw_root.is_absolute():
            raw_root = (
                repository_root
                / raw_root
            )

        output_directory = (
            args.output_directory
        )

        if not output_directory.is_absolute():
            output_directory = (
                repository_root
                / output_directory
            )

        required_files = (
            repository_root
            / "src"
            / "models"
            / "encoder.py",
            repository_root
            / "src"
            / "models"
            / "siamese_unet.py",
            repository_root
            / "src"
            / "data"
            / "oscd_dataset.py",
            repository_root
            / "tests"
            / "test_encoder_adaptation.py",
            repository_root
            / "tests"
            / "test_siamese_unet.py",
            repository_root
            / "tests"
            / "test_oscd_dataset.py",
        )

        missing_files = [
            str(path)
            for path in required_files
            if not path.is_file()
        ]

        if missing_files:
            raise Week3AuditError(
                "Required Week 3 files are missing:\n"
                + "\n".join(
                    f"  {path}"
                    for path in missing_files
                )
            )

        LOGGER.info(
            "Compiling source and tests."
        )

        compile_output = run_command(
            [
                sys.executable,
                "-m",
                "compileall",
                "-q",
                "src",
                "tests",
            ],
            repository_root=repository_root,
        )

        LOGGER.info(
            "Running Week 3 automated tests."
        )

        pytest_output = run_command(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests",
                "-q",
            ],
            repository_root=repository_root,
        )

        LOGGER.info(
            "Checking installed dependencies."
        )

        pip_check_output = run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "check",
            ],
            repository_root=repository_root,
        )

        LOGGER.info(
            "Checking Git whitespace."
        )

        git_diff_check_output = run_command(
            [
                "git",
                "diff",
                "--check",
            ],
            repository_root=repository_root,
        )

        git_status_output = run_command(
            [
                "git",
                "status",
                "--short",
            ],
            repository_root=repository_root,
        )

        LOGGER.info(
            "Loading one real OSCD training patch."
        )

        dataset = OSCDTrainingDataset(
            raw_root=raw_root,
            region_names=(
                args.region,
            ),
            band_names=(
                "B02",
                "B03",
                "B04",
                "B08",
            ),
            patch_size=64,
            stride=64,
        )

        sample = dataset[0]

        before = sample[
            "before"
        ].unsqueeze(
            0
        )
        after = sample[
            "after"
        ].unsqueeze(
            0
        )
        mask = sample[
            "mask"
        ].unsqueeze(
            0
        )

        expected_image_shape = (
            1,
            4,
            64,
            64,
        )
        expected_mask_shape = (
            1,
            1,
            64,
            64,
        )

        if tuple(before.shape) != expected_image_shape:
            raise Week3AuditError(
                f"Unexpected before shape: {tuple(before.shape)}"
            )

        if tuple(after.shape) != expected_image_shape:
            raise Week3AuditError(
                f"Unexpected after shape: {tuple(after.shape)}"
            )

        if tuple(mask.shape) != expected_mask_shape:
            raise Week3AuditError(
                f"Unexpected mask shape: {tuple(mask.shape)}"
            )

        mask_values = set(
            torch.unique(
                mask
            ).tolist()
        )

        if not mask_values.issubset(
            {
                0.0,
                1.0,
            }
        ):
            raise Week3AuditError(
                f"Training mask is not binary: {sorted(mask_values)}"
            )

        LOGGER.info(
            "Running real training patch through the model."
        )

        torch.manual_seed(
            42
        )

        model = SiameseUNet(
            input_channels=4,
            pretrained_encoder=False,
            decoder_channels=(
                64,
                32,
                16,
                16,
            ),
            head_channels=8,
            dropout_probability=0.0,
        )

        model.eval()

        with torch.inference_mode():
            logits = model(
                before,
                after,
            )
            swapped_logits = model(
                after,
                before,
            )

        expected_logit_shape = (
            1,
            1,
            64,
            64,
        )

        if tuple(logits.shape) != expected_logit_shape:
            raise Week3AuditError(
                f"Unexpected model output: {tuple(logits.shape)}"
            )

        if not torch.isfinite(
            logits
        ).all():
            raise Week3AuditError(
                "Model produced non-finite logits."
            )

        if not torch.allclose(
            logits,
            swapped_logits,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise Week3AuditError(
                "Model failed the real-data date-order symmetry check."
            )

        maximum_swap_error = float(
            torch.max(
                torch.abs(
                    logits
                    - swapped_logits
                )
            ).item()
        )

        total_parameters, trainable_parameters = (
            count_parameters(
                model
            )
        )

        completed_at = (
            datetime.now()
            .astimezone()
            .isoformat(
                timespec="seconds"
            )
        )

        report: dict[str, Any] = {
            "week": 3,
            "status": "complete",
            "completed_at": completed_at,
            "environment": {
                "python": (
                    f"{sys.version_info.major}."
                    f"{sys.version_info.minor}."
                    f"{sys.version_info.micro}"
                ),
                "torch": version(
                    "torch"
                ),
                "torchvision": version(
                    "torchvision"
                ),
                "pytest": version(
                    "pytest"
                ),
                "device": "cpu",
            },
            "checks": {
                "required_files": True,
                "compileall": True,
                "pytest": True,
                "pytest_summary": final_nonempty_line(
                    pytest_output
                ),
                "pip_check": True,
                "pip_check_summary": final_nonempty_line(
                    pip_check_output
                ),
                "git_diff_check": True,
                "training_only_dataset": True,
                "test_labels_requested": False,
                "real_patch_model_forward": True,
                "date_order_symmetry": True,
            },
            "dataset": {
                "region": sample[
                    "region"
                ],
                "patch_id": sample[
                    "patch_id"
                ],
                "before_shape": list(
                    before.shape
                ),
                "after_shape": list(
                    after.shape
                ),
                "mask_shape": list(
                    mask.shape
                ),
                "mask_values": sorted(
                    mask_values
                ),
                "change_fraction": float(
                    mask.mean().item()
                ),
            },
            "model": {
                "input_channels": 4,
                "output_shape": list(
                    logits.shape
                ),
                "raw_logits": True,
                "shared_encoder": True,
                "difference_fusion": "absolute",
                "maximum_swap_error": maximum_swap_error,
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
            },
            "repository_status": (
                git_status_output.splitlines()
            ),
            "compile_output": compile_output,
            "git_diff_check_output": git_diff_check_output,
        }

        json_path = (
            output_directory
            / "week3_completion_report.json"
        )
        markdown_path = (
            output_directory
            / "week3_completion_report.md"
        )

        write_json_atomic(
            path=json_path,
            payload=report,
        )

        markdown = f"""# GeoWatch Week 3 Completion Report

**Status:** Complete  
**Completed:** {completed_at}

## Architecture

- Weight-shared Siamese ResNet-18 encoder
- Four-band and six-band Sentinel-2 support
- ImageNet first-convolution adaptation
- Absolute-difference fusion at five feature scales
- GroupNorm U-Net decoder
- Full-resolution one-channel raw logits
- Date-order symmetry verified

## Dataset

- Dataset: OSCD official training regions
- Audit region: `{sample["region"]}`
- Audit patch: `{sample["patch_id"]}`
- Before shape: `{tuple(before.shape)}`
- After shape: `{tuple(after.shape)}`
- Mask shape: `{tuple(mask.shape)}`
- Binary mask values: `{sorted(mask_values)}`
- Patch change fraction: `{float(mask.mean().item()):.8f}`
- OSCD test labels requested: **No**

## Model integration

- Output shape: `{tuple(logits.shape)}`
- Output values finite: **Yes**
- Shared encoder: **Yes**
- Date-order symmetric: **Yes**
- Maximum swap error: `{maximum_swap_error}`
- Total parameters in audit model: `{total_parameters:,}`
- Trainable parameters: `{trainable_parameters:,}`

## Verification

- Required files: Passed
- Python compilation: Passed
- Automated tests: `{final_nonempty_line(pytest_output)}`
- Dependency consistency: `{final_nonempty_line(pip_check_output)}`
- Git whitespace check: Passed

## Evaluation discipline

No OSCD test-label path was supplied to the dataset or model audit.
Week 3 performed architecture and training-data validation only. Quantitative
benchmark comparison remains governed by the frozen Week 2 OSCD baseline and
the later held-out evaluation protocol.
"""

        write_text_atomic(
            path=markdown_path,
            content=markdown,
        )

        print(
            "GeoWatch Week 3 final audit passed"
        )
        print(
            "  Test summary:",
            final_nonempty_line(
                pytest_output
            ),
        )
        print(
            "  Dependency check:",
            final_nonempty_line(
                pip_check_output
            ),
        )
        print(
            "  Training region:",
            sample["region"],
        )
        print(
            "  Input shape:",
            tuple(before.shape),
        )
        print(
            "  Mask shape:",
            tuple(mask.shape),
        )
        print(
            "  Logit shape:",
            tuple(logits.shape),
        )
        print(
            "  Maximum swap error:",
            maximum_swap_error,
        )
        print(
            "  Test labels requested:",
            False,
        )
        print(
            "  JSON report:",
            json_path.relative_to(
                repository_root
            ),
        )
        print(
            "  Markdown report:",
            markdown_path.relative_to(
                repository_root
            ),
        )

        return 0

    except (
        FileNotFoundError,
        PermissionError,
        Week3AuditError,
        RuntimeError,
        ValueError,
        TypeError,
        OSError,
    ) as error:
        LOGGER.error(
            "%s",
            error,
        )
        return 1

    except Exception:
        LOGGER.exception(
            "Unexpected Week 3 audit failure."
        )
        return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )
