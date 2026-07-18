from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


FROZEN_THRESHOLD = 0.76
FROZEN_CHECKPOINT_EPOCH = 24
FROZEN_CHECKPOINT_SHA256 = (
    "61e53ba86bc108d6ccbbb636c8da88c967e07da41471504878c74df6a903ea94"
)
FROZEN_TEST_RESULT_SHA256 = (
    "e45e6265608707e3d1e439737bca59e9fde10010dd8c73cca420d7881bb9a8f9"
)
FROZEN_FAILURE_ANALYSIS_SHA256 = (
    "077bd0c56d200f3e7cc14019ab49b832aed020dfa72342908f4f7a6539e024d9"
)
EXPECTED_TEST_PIXELS = 3_077_936
EXPECTED_TEST_REGIONS = 10
EXPECTED_TEST_PATCHES = 69
EXPECTED_GALLERY_IMAGES = 12

HYDERABAD_REQUIRED_FILES = (
    "before/B02.tif",
    "before/B03.tif",
    "before/B04.tif",
    "before/B08.tif",
    "after/B02.tif",
    "after/B03.tif",
    "after/B04.tif",
    "after/B08.tif",
)


@dataclass(frozen=True)
class IntegrityRecord:
    path: str
    sha256: str
    expected_sha256: str
    valid: bool


@dataclass(frozen=True)
class HyderabadRecord:
    status: str
    input_root: str
    present_files: tuple[str, ...]
    missing_files: tuple[str, ...]
    audit_path: str | None
    qualitative_path: str | None


def calculate_sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as source:
        while True:
            block = source.read(
                1024
                * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            f"Expected a JSON object: {path}"
        )

    return payload


def validate_integrity(
    path: Path,
    expected_sha256: str,
) -> IntegrityRecord:
    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    actual = calculate_sha256(
        path
    )

    if actual != expected_sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: "
            f"{actual} versus {expected_sha256}"
        )

    return IntegrityRecord(
        path=str(
            path
        ),
        sha256=actual,
        expected_sha256=expected_sha256,
        valid=True,
    )


def validate_test_results(
    payload: dict[str, Any],
) -> None:
    protocol = payload[
        "protocol"
    ]

    dataset = payload[
        "dataset"
    ]

    evaluation = payload[
        "evaluation"
    ]

    access = payload[
        "access"
    ]

    micro = evaluation[
        "micro"
    ]

    if float(
        protocol[
            "threshold"
        ]
    ) != FROZEN_THRESHOLD:
        raise RuntimeError(
            "Frozen threshold mismatch."
        )

    if int(
        protocol[
            "checkpoint_epoch"
        ]
    ) != FROZEN_CHECKPOINT_EPOCH:
        raise RuntimeError(
            "Frozen checkpoint epoch mismatch."
        )

    if str(
        protocol[
            "checkpoint_sha256"
        ]
    ) != FROZEN_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Frozen checkpoint SHA-256 mismatch."
        )

    if int(
        dataset[
            "region_count"
        ]
    ) != EXPECTED_TEST_REGIONS:
        raise RuntimeError(
            "Official test-region count mismatch."
        )

    if int(
        dataset[
            "patch_count"
        ]
    ) != EXPECTED_TEST_PATCHES:
        raise RuntimeError(
            "Official test-patch count mismatch."
        )

    if int(
        dataset[
            "valid_pixel_count"
        ]
    ) != EXPECTED_TEST_PIXELS:
        raise RuntimeError(
            "Official valid-pixel count mismatch."
        )

    if int(
        micro[
            "evaluated_pixels"
        ]
    ) != EXPECTED_TEST_PIXELS:
        raise RuntimeError(
            "Official evaluated-pixel count mismatch."
        )

    if access.get(
        "test_data_used_for_tuning"
    ) is not False:
        raise RuntimeError(
            "Official test data must not be used for tuning."
        )

    if access.get(
        "threshold_modified"
    ) is not False:
        raise RuntimeError(
            "Frozen threshold was modified."
        )

    if access.get(
        "checkpoint_modified"
    ) is not False:
        raise RuntimeError(
            "Frozen checkpoint was modified."
        )


def validate_failure_analysis(
    payload: dict[str, Any],
    test_result_sha256: str,
) -> None:
    integrity = payload[
        "integrity"
    ]

    regions = payload[
        "regions"
    ]

    if str(
        integrity[
            "test_result_sha256"
        ]
    ) != test_result_sha256:
        raise RuntimeError(
            "Failure analysis references another test result."
        )

    if float(
        integrity[
            "threshold"
        ]
    ) != FROZEN_THRESHOLD:
        raise RuntimeError(
            "Failure-analysis threshold mismatch."
        )

    if int(
        integrity[
            "checkpoint_epoch"
        ]
    ) != FROZEN_CHECKPOINT_EPOCH:
        raise RuntimeError(
            "Failure-analysis checkpoint epoch mismatch."
        )

    if integrity.get(
        "test_data_used_for_tuning"
    ) is not False:
        raise RuntimeError(
            "Failure analysis indicates test-based tuning."
        )

    if len(
        regions
    ) != EXPECTED_TEST_REGIONS:
        raise RuntimeError(
            "Failure-analysis region count mismatch."
        )


def validate_gallery(
    gallery_root: Path,
    test_result_sha256: str,
) -> tuple[dict[str, Any], tuple[IntegrityRecord, ...]]:
    manifest_path = (
        gallery_root
        / "manifest.json"
    )

    manifest = load_json(
        manifest_path
    )

    protocol = manifest[
        "protocol"
    ]

    generation = manifest[
        "generation"
    ]

    entries = manifest[
        "entries"
    ]

    if str(
        protocol[
            "test_result_sha256"
        ]
    ) != test_result_sha256:
        raise RuntimeError(
            "Gallery references another test result."
        )

    if float(
        protocol[
            "threshold"
        ]
    ) != FROZEN_THRESHOLD:
        raise RuntimeError(
            "Gallery threshold mismatch."
        )

    if protocol.get(
        "test_data_used_for_tuning"
    ) is not False:
        raise RuntimeError(
            "Gallery indicates test-based tuning."
        )

    if int(
        generation[
            "inferred_patches"
        ]
    ) != EXPECTED_TEST_PATCHES:
        raise RuntimeError(
            "Gallery inference patch count mismatch."
        )

    if int(
        generation[
            "gallery_entry_count"
        ]
    ) != EXPECTED_GALLERY_IMAGES:
        raise RuntimeError(
            "Gallery image count mismatch."
        )

    if len(
        entries
    ) != EXPECTED_GALLERY_IMAGES:
        raise RuntimeError(
            "Gallery manifest-entry count mismatch."
        )

    records: list[
        IntegrityRecord
    ] = []

    for entry in entries:
        image_path = (
            gallery_root
            / str(
                entry[
                    "file"
                ]
            )
        )

        records.append(
            validate_integrity(
                image_path,
                str(
                    entry[
                        "file_sha256"
                    ]
                ),
            )
        )

    return (
        manifest,
        tuple(
            records
        ),
    )


def inspect_hyderabad(
    project_root: Path,
) -> HyderabadRecord:
    input_root = (
        project_root
        / "data"
        / "qualitative"
        / "hyderabad"
    )

    present: list[
        str
    ] = []

    missing: list[
        str
    ] = []

    for relative in HYDERABAD_REQUIRED_FILES:
        path = (
            input_root
            / relative
        )

        if path.is_file() and path.stat().st_size > 0:
            present.append(
                relative
            )
        else:
            missing.append(
                relative
            )

    audit_path = (
        project_root
        / "reports"
        / "week6"
        / "hyderabad_input_audit.json"
    )

    qualitative_path = (
        project_root
        / "reports"
        / "week6"
        / "hyderabad_qualitative.json"
    )

    if missing:
        status = "pending_input"
    elif qualitative_path.is_file():
        status = "qualitative_complete"
    elif audit_path.is_file():
        status = "input_audited"
    else:
        status = "input_ready"

    return HyderabadRecord(
        status=status,
        input_root=str(
            input_root
        ),
        present_files=tuple(
            present
        ),
        missing_files=tuple(
            missing
        ),
        audit_path=(
            str(
                audit_path
            )
            if audit_path.is_file()
            else None
        ),
        qualitative_path=(
            str(
                qualitative_path
            )
            if qualitative_path.is_file()
            else None
        ),
    )


def read_review_status(
    review_path: Path,
) -> dict[str, int]:
    if not review_path.is_file():
        return {
            "rows": 0,
            "reviewed": 0,
            "unreviewed": 0,
        }

    with review_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as source:
        rows = list(
            csv.DictReader(
                source
            )
        )

    reviewed = sum(
        1
        for row in rows
        if str(
            row.get(
                "visual_failure_mode",
                "",
            )
        ).strip()
    )

    return {
        "rows": len(
            rows
        ),
        "reviewed": reviewed,
        "unreviewed": (
            len(
                rows
            )
            - reviewed
        ),
    }


def format_float(
    value: Any,
) -> str:
    return f"{float(value):.6f}"


def build_region_table(
    regions: list[dict[str, Any]],
) -> str:
    ranked = sorted(
        regions,
        key=lambda region: float(
            region[
                "f1"
            ]
        ),
        reverse=True,
    )

    lines = [
        "| Rank | Region | Precision | Recall | F1 | IoU | Prevalence | Prediction fraction |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]

    for rank, region in enumerate(
        ranked,
        start=1,
    ):
        lines.append(
            "| "
            + " | ".join(
                (
                    str(
                        rank
                    ),
                    str(
                        region[
                            "region"
                        ]
                    ),
                    format_float(
                        region[
                            "precision"
                        ]
                    ),
                    format_float(
                        region[
                            "recall"
                        ]
                    ),
                    format_float(
                        region[
                            "f1"
                        ]
                    ),
                    format_float(
                        region[
                            "iou"
                        ]
                    ),
                    format_float(
                        region[
                            "change_prevalence"
                        ]
                    ),
                    format_float(
                        region[
                            "predicted_change_fraction"
                        ]
                    ),
                )
            )
            + " |"
        )

    return "\n".join(
        lines
    )


def build_report(
    test_payload: dict[str, Any],
    failure_payload: dict[str, Any],
    gallery_manifest: dict[str, Any],
    hyderabad: HyderabadRecord,
    review_status: dict[str, int],
    test_integrity: IntegrityRecord,
    failure_integrity: IntegrityRecord,
    checkpoint_integrity: IntegrityRecord,
    test_command_count: int,
) -> str:
    protocol = test_payload[
        "protocol"
    ]

    evaluation = test_payload[
        "evaluation"
    ]

    micro = evaluation[
        "micro"
    ]

    macro = evaluation[
        "macro"
    ]

    generalization = failure_payload[
        "generalization"
    ]

    deltas = generalization[
        "delta_test_minus_validation"
    ]

    summary = failure_payload[
        "summary"
    ]

    regions = evaluation[
        "per_region"
    ]

    gallery_generation = gallery_manifest[
        "generation"
    ]

    hyderabad_note = {
        "pending_input": (
            "Not executed because the eight aligned Hyderabad Sentinel-2 "
            "GeoTIFF bands are not available. No substitute city or synthetic "
            "result was used."
        ),
        "input_ready": (
            "All required Hyderabad bands are present. Metadata auditing and "
            "qualitative inference remain pending."
        ),
        "input_audited": (
            "Hyderabad inputs passed metadata auditing. Qualitative model "
            "inference remains pending."
        ),
        "qualitative_complete": (
            "Hyderabad qualitative inference completed without labels and "
            "without changing official test metrics."
        ),
    }[
        hyderabad.status
    ]

    lines = [
        "# GeoWatch Week 6 Evaluation Report",
        "",
        "## Completion status",
        "",
        "The frozen official OSCD evaluation, quantitative failure analysis, "
        "and failure gallery are complete. Hyderabad evaluation is treated as "
        "an optional external qualitative demonstration and is never used for "
        "model selection or reported as labelled test performance.",
        "",
        f"- Week 6 status: `{hyderabad.status}`",
        f"- Official evaluation invocations recorded: `{test_command_count}`",
        "- Test-based tuning: `false`",
        "- Threshold retuning after test access: `false`",
        "",
        "## Frozen protocol",
        "",
        f"- Checkpoint epoch: `{protocol['checkpoint_epoch']}`",
        f"- Checkpoint SHA-256: `{protocol['checkpoint_sha256']}`",
        f"- Probability threshold: `{protocol['threshold']}`",
        f"- Bands: `{', '.join(protocol['bands'])}`",
        f"- Patch size: `{protocol['patch_size']}`",
        f"- Stride: `{protocol['stride']}`",
        f"- Official test regions: `{len(protocol['test_regions'])}`",
        f"- Official evaluated pixels: `{micro['evaluated_pixels']}`",
        "",
        "## Official test results",
        "",
        "| Aggregation | Precision | Recall | F1 | IoU | Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
        (
            "| Micro | "
            f"{format_float(micro['precision'])} | "
            f"{format_float(micro['recall'])} | "
            f"{format_float(micro['f1'])} | "
            f"{format_float(micro['iou'])} | "
            f"{format_float(micro['accuracy'])} |"
        ),
        (
            "| Macro mean | "
            f"{format_float(macro['mean']['precision'])} | "
            f"{format_float(macro['mean']['recall'])} | "
            f"{format_float(macro['mean']['f1'])} | "
            f"{format_float(macro['mean']['iou'])} | "
            f"{format_float(macro['mean']['accuracy'])} |"
        ),
        (
            "| Macro median | "
            f"{format_float(macro['median']['precision'])} | "
            f"{format_float(macro['median']['recall'])} | "
            f"{format_float(macro['median']['f1'])} | "
            f"{format_float(macro['median']['iou'])} | "
            f"{format_float(macro['median']['accuracy'])} |"
        ),
        "",
        f"- Change prevalence: `{format_float(micro['change_prevalence'])}`",
        (
            "- Predicted change fraction: "
            f"`{format_float(micro['predicted_change_fraction'])}`"
        ),
        "",
        "## Validation-to-test generalization",
        "",
        (
            "- Precision delta: "
            f"`{format_float(deltas['precision'])}`"
        ),
        (
            "- Recall delta: "
            f"`{format_float(deltas['recall'])}`"
        ),
        f"- F1 delta: `{format_float(deltas['f1'])}`",
        f"- IoU delta: `{format_float(deltas['iou'])}`",
        (
            "- Micro minus macro-mean F1: "
            f"`{format_float(failure_payload['distribution']['micro_minus_macro_mean_f1'])}`"
        ),
        "",
        "The higher micro score than macro-mean score indicates substantial "
        "geographic variability. Performance is therefore reported with both "
        "global pixel aggregation and equal-weight regional aggregation.",
        "",
        "## Per-region results",
        "",
        build_region_table(
            regions
        ),
        "",
        "## Failure analysis",
        "",
        f"- Strongest region: `{summary['strongest_region']}`",
        f"- Weakest region: `{summary['weakest_region']}`",
        (
            "- False-positive focus regions: `"
            + ", ".join(
                summary[
                    "false_positive_focus_regions"
                ]
            )
            + "`"
        ),
        (
            "- False-negative focus regions: `"
            + ", ".join(
                summary[
                    "false_negative_focus_regions"
                ]
            )
            + "`"
        ),
        (
            "- Gallery focus regions: `"
            + ", ".join(
                gallery_generation[
                    "focus_regions"
                ]
            )
            + "`"
        ),
        f"- Gallery images: `{gallery_generation['gallery_entry_count']}`",
        f"- Gallery review rows: `{review_status['rows']}`",
        f"- Gallery rows classified: `{review_status['reviewed']}`",
        f"- Gallery rows pending review: `{review_status['unreviewed']}`",
        "",
        "Failure-gallery selection is post-hoc diagnostic analysis only. It "
        "does not alter the checkpoint, threshold, preprocessing, or official "
        "metrics.",
        "",
        "## Hyderabad qualitative status",
        "",
        hyderabad_note,
        "",
        f"- Input root: `{hyderabad.input_root}`",
        f"- Present required files: `{len(hyderabad.present_files)}`",
        f"- Missing required files: `{len(hyderabad.missing_files)}`",
        "",
        "## Reproducibility and integrity",
        "",
        f"- Checkpoint SHA-256 verified: `{checkpoint_integrity.sha256}`",
        f"- Test result SHA-256 verified: `{test_integrity.sha256}`",
        f"- Failure analysis SHA-256 verified: `{failure_integrity.sha256}`",
        f"- Gallery entries verified: `{len(gallery_manifest['entries'])}`",
        "- Official result overwrite protection: `enabled`",
        "- Official test data used for tuning: `false`",
        "",
        "## Limitations",
        "",
        "- Regional performance varies substantially.",
        "- Several low-prevalence regions show strong false-positive behaviour.",
        "- Accuracy is secondary because unchanged pixels dominate the dataset.",
        "- The Hyderabad external demonstration cannot be claimed until genuine "
        "aligned imagery is available.",
        "- Repeated execution produced identical console metrics, but only the "
        "currently hashed JSON artifact is treated as the frozen result.",
        "",
    ]

    return "\n".join(
        lines
    )


def run_tests(
    project_root: Path,
    python_executable: str,
) -> dict[str, Any]:
    command = [
        python_executable,
        "-m",
        "pytest",
        "tests",
        "-q",
        "-W",
        "error",
    ]

    completed = subprocess.run(
        command,
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "Week 6 acceptance tests failed.\n"
            + completed.stdout
            + completed.stderr
        )

    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def atomic_write_text(
    path: Path,
    content: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name,
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(
            content
        )

        temporary_path = Path(
            temporary.name
        )

    temporary_path.replace(
        path
    )


def build_argument_parser(
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(
            "."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/week6/week6_evaluation_report.md"
        ),
    )

    parser.add_argument(
        "--completion-json",
        type=Path,
        default=Path(
            "reports/week6/week6_completion.json"
        ),
    )

    parser.add_argument(
        "--run-tests",
        action="store_true",
    )

    parser.add_argument(
        "--python",
        default=sys.executable,
    )

    return parser


def main(
) -> int:
    arguments = build_argument_parser().parse_args()

    project_root = arguments.project_root.resolve()

    checkpoint_path = (
        project_root
        / "experiments"
        / "run_full"
        / "checkpoints"
        / "best_model_epoch24.pt"
    )

    test_results_path = (
        project_root
        / "reports"
        / "week6"
        / "test_results.json"
    )

    failure_path = (
        project_root
        / "reports"
        / "week6"
        / "failure_analysis.json"
    )

    gallery_root = (
        project_root
        / "reports"
        / "week6"
        / "failure_gallery"
    )

    review_path = (
        gallery_root
        / "gallery_review.csv"
    )

    checkpoint_integrity = validate_integrity(
        checkpoint_path,
        FROZEN_CHECKPOINT_SHA256,
    )

    test_integrity = validate_integrity(
        test_results_path,
        FROZEN_TEST_RESULT_SHA256,
    )

    failure_integrity = validate_integrity(
        failure_path,
        FROZEN_FAILURE_ANALYSIS_SHA256,
    )

    test_payload = load_json(
        test_results_path
    )

    failure_payload = load_json(
        failure_path
    )

    validate_test_results(
        test_payload
    )

    validate_failure_analysis(
        failure_payload,
        test_integrity.sha256,
    )

    gallery_manifest, gallery_integrity = validate_gallery(
        gallery_root,
        test_integrity.sha256,
    )

    hyderabad = inspect_hyderabad(
        project_root
    )

    review_status = read_review_status(
        review_path
    )

    test_run = (
        run_tests(
            project_root,
            arguments.python,
        )
        if arguments.run_tests
        else None
    )

    test_command_count = int(
        failure_payload[
            "integrity"
        ][
            "reported_full_evaluation_invocations"
        ]
    )

    report = build_report(
        test_payload=test_payload,
        failure_payload=failure_payload,
        gallery_manifest=gallery_manifest,
        hyderabad=hyderabad,
        review_status=review_status,
        test_integrity=test_integrity,
        failure_integrity=failure_integrity,
        checkpoint_integrity=checkpoint_integrity,
        test_command_count=test_command_count,
    )

    output_path = (
        project_root
        / arguments.output
    )

    completion_path = (
        project_root
        / arguments.completion_json
    )

    atomic_write_text(
        output_path,
        report.rstrip()
        + "\n",
    )

    completion_status = (
        "complete"
        if hyderabad.status
        == "qualitative_complete"
        else "complete_with_optional_hyderabad_pending"
    )

    completion = {
        "status": completion_status,
        "frozen_protocol": {
            "checkpoint_epoch": FROZEN_CHECKPOINT_EPOCH,
            "checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
            "threshold": FROZEN_THRESHOLD,
            "test_result_sha256": test_integrity.sha256,
            "failure_analysis_sha256": failure_integrity.sha256,
        },
        "official_test": {
            "regions": EXPECTED_TEST_REGIONS,
            "patches": EXPECTED_TEST_PATCHES,
            "evaluated_pixels": EXPECTED_TEST_PIXELS,
            "micro": test_payload[
                "evaluation"
            ][
                "micro"
            ],
            "macro": test_payload[
                "evaluation"
            ][
                "macro"
            ],
        },
        "failure_analysis": {
            "strongest_region": failure_payload[
                "summary"
            ][
                "strongest_region"
            ],
            "weakest_region": failure_payload[
                "summary"
            ][
                "weakest_region"
            ],
            "gallery_images": len(
                gallery_integrity
            ),
            "gallery_review": review_status,
        },
        "hyderabad": asdict(
            hyderabad
        ),
        "tests": test_run,
        "report": str(
            output_path
        ),
        "test_data_used_for_tuning": False,
        "threshold_modified": False,
        "checkpoint_modified": False,
    }

    atomic_write_text(
        completion_path,
        json.dumps(
            completion,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )

    validate_integrity(
        checkpoint_path,
        FROZEN_CHECKPOINT_SHA256,
    )

    validate_integrity(
        test_results_path,
        FROZEN_TEST_RESULT_SHA256,
    )

    validate_integrity(
        failure_path,
        FROZEN_FAILURE_ANALYSIS_SHA256,
    )

    print("GeoWatch Week 6 finalization completed")
    print("  Status:", completion_status)
    print("  Report:", output_path)
    print("  Completion record:", completion_path)
    print("  Official test result unchanged:", True)
    print("  Threshold modified:", False)
    print("  Checkpoint modified:", False)
    print("  Hyderabad status:", hyderabad.status)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
