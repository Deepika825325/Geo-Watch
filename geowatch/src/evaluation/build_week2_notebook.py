"""Build the GeoWatch Week 2 EDA and baseline notebook.

The notebook is generated from validated Week 2 JSON, CSV and image
artifacts. It does not recompute or retune either classical baseline.

Quantitative results are restricted to labelled OSCD regions. GeoWatch's
unlabelled Hyderabad AOI is excluded from all metric tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any, Mapping

import nbformat
from nbformat import NotebookNode
from nbformat.v4 import (
    new_code_cell,
    new_markdown_cell,
    new_notebook,
)


LOGGER = logging.getLogger(
    "geowatch.build_week2_notebook"
)


class NotebookBuildError(RuntimeError):
    """Raised when the Week 2 notebook cannot be built safely."""


def load_json(
    path: Path,
) -> Mapping[str, Any]:
    """Load and validate one JSON object."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Required JSON file does not exist: {path}"
        )

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )
    except json.JSONDecodeError as error:
        raise NotebookBuildError(
            f"Invalid JSON file: {path}"
        ) from error

    if not isinstance(payload, Mapping):
        raise NotebookBuildError(
            f"JSON root must be an object: {path}"
        )

    return payload


def load_csv(
    path: Path,
) -> list[dict[str, str]]:
    """Load a CSV file as dictionaries."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Required CSV file does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    if not rows:
        raise NotebookBuildError(
            f"CSV file contains no rows: {path}"
        )

    return rows


def validate_artifacts(
    eda: Mapping[str, Any],
    otsu: Mapping[str, Any],
    cva: Mapping[str, Any],
    otsu_rows: list[dict[str, str]],
    cva_rows: list[dict[str, str]],
) -> None:
    """Validate the artifacts used to construct the notebook."""
    if eda.get("status") != "success":
        raise NotebookBuildError(
            "OSCD EDA status is not success."
        )

    if otsu.get("status") != "success":
        raise NotebookBuildError(
            "Otsu baseline status is not success."
        )

    if cva.get("status") != "success":
        raise NotebookBuildError(
            "CVA baseline status is not success."
        )

    expected_counts = {
        "train": 14,
        "test": 10,
        "overall": 24,
    }

    for name, artifact in (
        ("EDA", eda),
        ("Otsu", otsu),
        ("CVA", cva),
    ):
        counts = artifact.get("region_counts")

        if counts != expected_counts:
            raise NotebookBuildError(
                f"{name} has unexpected region counts: {counts}"
            )

        if (
            artifact.get(
                "custom_hyderabad_aoi_used_for_metrics"
            )
            is not False
        ):
            raise NotebookBuildError(
                f"{name} did not explicitly exclude Hyderabad "
                "from metrics."
            )

    if len(otsu_rows) != 24:
        raise NotebookBuildError(
            f"Expected 24 Otsu region rows; found {len(otsu_rows)}."
        )

    if len(cva_rows) != 24:
        raise NotebookBuildError(
            f"Expected 24 CVA region rows; found {len(cva_rows)}."
        )


def select_representative_region(
    otsu_rows: list[dict[str, str]],
) -> str:
    """Select the median-F1 test region without cherry-picking."""
    test_rows = [
        row
        for row in otsu_rows
        if row.get("split") == "test"
    ]

    if len(test_rows) != 10:
        raise NotebookBuildError(
            f"Expected 10 Otsu test rows; found {len(test_rows)}."
        )

    ordered = sorted(
        test_rows,
        key=lambda row: (
            float(row["f1_score"]),
            row["region"],
        ),
    )

    return ordered[
        len(ordered) // 2
    ]["region"]


def build_notebook(
    eda: Mapping[str, Any],
    otsu: Mapping[str, Any],
    cva: Mapping[str, Any],
    representative_region: str,
) -> NotebookNode:
    """Construct the complete Week 2 notebook."""
    eda_overall = eda["overall"]
    eda_train = eda["train"]
    eda_test = eda["test"]

    otsu_test = otsu["test_metrics"]
    cva_test = cva["test_metrics"]

    strongest_name = (
        "Band Difference + Otsu"
        if otsu_test["f1_score"]
        >= cva_test["f1_score"]
        else "CVA + PCA + K-Means"
    )

    strongest_metrics = (
        otsu_test
        if strongest_name
        == "Band Difference + Otsu"
        else cva_test
    )

    f1_relative_improvement = (
        (
            otsu_test["f1_score"]
            - cva_test["f1_score"]
        )
        / cva_test["f1_score"]
    )

    iou_relative_improvement = (
        (
            otsu_test["iou"]
            - cva_test["iou"]
        )
        / cva_test["iou"]
    )

    cells = [
        new_markdown_cell(
            dedent(
                f"""
                # GeoWatch Week 2 — EDA and Classical Baselines

                **Objective:** establish transparent non-deep-learning
                change-detection baselines before training the Siamese U-Net.

                **Quantitative dataset:** OSCD only.

                **Custom Hyderabad AOI:** qualitative demonstration only;
                it is not used in any metric calculation.

                **Official OSCD split:** 14 training regions and 10 testing
                regions.
                """
            ).strip()
        ),
        new_markdown_cell(
            dedent(
                """
                ## Evaluation protocol

                The positive change class is evaluated using precision,
                recall, F1 and Intersection over Union.

                Overall pixel accuracy is not used as the headline metric.
                OSCD is highly imbalanced, so predicting unchanged for most
                pixels can produce apparently high accuracy while missing
                real change.

                Baseline fitting rules:

                - Otsu threshold: fitted from OSCD training imagery only.
                - StandardScaler, PCA and K-Means: fitted from OSCD training
                  imagery only.
                - Test imagery and labels are excluded from fitting.
                - Ground-truth labels are used only for quantitative
                  evaluation.
                """
            ).strip()
        ),
        new_code_cell(
            dedent(
                """
                from pathlib import Path
                import csv
                import json

                import matplotlib.pyplot as plt
                import numpy as np
                from PIL import Image
                from IPython.display import Markdown, display


                ROOT = Path.cwd()

                if not (ROOT / "reports").is_dir():
                    ROOT = ROOT.parent

                assert (ROOT / "reports").is_dir(), (
                    "Run this notebook from the repository root "
                    "or from the notebooks directory."
                )

                EDA_DIR = ROOT / "reports/week2/eda"
                OTSU_DIR = (
                    ROOT
                    / "reports/week2/baselines/band_diff_otsu"
                )
                CVA_DIR = (
                    ROOT
                    / "reports/week2/baselines/cva_pca_kmeans"
                )

                eda = json.loads(
                    (
                        EDA_DIR
                        / "oscd_dataset_statistics.json"
                    ).read_text(encoding="utf-8")
                )
                otsu = json.loads(
                    (
                        OTSU_DIR
                        / "band_diff_otsu_report.json"
                    ).read_text(encoding="utf-8")
                )
                cva = json.loads(
                    (
                        CVA_DIR
                        / "cva_pca_kmeans_report.json"
                    ).read_text(encoding="utf-8")
                )

                print("Week 2 artifacts loaded")
                print("Repository root:", ROOT)
                """
            ).strip()
        ),
        new_markdown_cell(
            "## OSCD dataset characteristics"
        ),
        new_code_cell(
            dedent(
                f"""
                dataset_table = '''
                | Split | Regions | Change pixels | Unchanged:change |
                |---|---:|---:|---:|
                | Train | 14 | {eda_train['pixel_weighted_change_fraction']:.4%} | {eda_train['unchanged_to_change_ratio']:.2f}:1 |
                | Test | 10 | {eda_test['pixel_weighted_change_fraction']:.4%} | {eda_test['unchanged_to_change_ratio']:.2f}:1 |
                | Overall | 24 | {eda_overall['pixel_weighted_change_fraction']:.4%} | {eda_overall['unchanged_to_change_ratio']:.2f}:1 |
                '''

                display(Markdown(dataset_table))

                print(
                    "Only",
                    f"{eda_overall['pixel_weighted_change_fraction']:.2%}",
                    "of evaluated OSCD pixels belong to the change class."
                )
                """
            ).strip()
        ),
        new_code_cell(
            dedent(
                """
                figure_paths = [
                    (
                        EDA_DIR
                        / "oscd_change_fraction_by_region.png",
                        "Change prevalence by region",
                    ),
                    (
                        EDA_DIR
                        / "oscd_representative_examples.png",
                        "Representative before/after examples",
                    ),
                ]

                for figure_path, title in figure_paths:
                    assert figure_path.is_file(), figure_path

                    image = np.asarray(
                        Image.open(figure_path).convert("RGB")
                    )

                    plt.figure(figsize=(12, 6))
                    plt.imshow(image)
                    plt.title(title)
                    plt.axis("off")
                    plt.tight_layout()
                    plt.show()
                """
            ).strip()
        ),
        new_markdown_cell(
            "## Classical baseline results"
        ),
        new_code_cell(
            dedent(
                f"""
                result_table = '''
                | Baseline | Precision | Recall | F1 | IoU | Accuracy* |
                |---|---:|---:|---:|---:|---:|
                | Band Difference + Otsu | {otsu_test['precision']:.6f} | {otsu_test['recall']:.6f} | {otsu_test['f1_score']:.6f} | {otsu_test['iou']:.6f} | {otsu_test['accuracy']:.6f} |
                | CVA + PCA + K-Means | {cva_test['precision']:.6f} | {cva_test['recall']:.6f} | {cva_test['f1_score']:.6f} | {cva_test['iou']:.6f} | {cva_test['accuracy']:.6f} |
                '''

                display(Markdown(result_table))
                display(
                    Markdown(
                        "\\\\*Accuracy is a secondary diagnostic because "
                        "unchanged pixels dominate OSCD."
                    )
                )
                """
            ).strip()
        ),
        new_code_cell(
            dedent(
                """
                metric_names = [
                    "precision",
                    "recall",
                    "f1_score",
                    "iou",
                ]

                display_names = [
                    "Precision",
                    "Recall",
                    "F1",
                    "IoU",
                ]

                otsu_values = [
                    otsu["test_metrics"][metric]
                    for metric in metric_names
                ]
                cva_values = [
                    cva["test_metrics"][metric]
                    for metric in metric_names
                ]

                positions = np.arange(
                    len(metric_names)
                )
                width = 0.36

                plt.figure(figsize=(9, 5.5))
                plt.bar(
                    positions - width / 2,
                    otsu_values,
                    width,
                    label="Band Difference + Otsu",
                )
                plt.bar(
                    positions + width / 2,
                    cva_values,
                    width,
                    label="CVA + PCA + K-Means",
                )
                plt.xticks(
                    positions,
                    display_names,
                )
                plt.ylim(0.0, 1.0)
                plt.ylabel("Score")
                plt.title(
                    "OSCD test performance — change class"
                )
                plt.legend()
                plt.grid(
                    axis="y",
                    alpha=0.3,
                )
                plt.tight_layout()
                plt.show()
                """
            ).strip()
        ),
        new_markdown_cell(
            dedent(
                f"""
                ## Qualitative comparison

                The example below is **{representative_region}**, selected
                deterministically as the median-F1 Otsu test region. It is
                not manually selected as a best-looking result.

                Difference and magnitude images are diagnostic intensity
                maps. Binary predictions are compared with the OSCD
                ground-truth mask.
                """
            ).strip()
        ),
        new_code_cell(
            dedent(
                f"""
                region = "{representative_region}"

                otsu_preview_path = (
                    OTSU_DIR
                    / "difference_previews/test"
                    / f"{{region}}.png"
                )
                otsu_prediction_path = (
                    OTSU_DIR
                    / "predictions/test"
                    / f"{{region}}.png"
                )
                cva_preview_path = (
                    CVA_DIR
                    / "magnitude_previews/test"
                    / f"{{region}}.png"
                )
                cva_prediction_path = (
                    CVA_DIR
                    / "predictions/test"
                    / f"{{region}}.png"
                )

                with (
                    EDA_DIR
                    / "oscd_region_statistics.csv"
                ).open(
                    "r",
                    encoding="utf-8-sig",
                    newline="",
                ) as file:
                    eda_rows = list(
                        csv.DictReader(file)
                    )

                region_row = next(
                    row
                    for row in eda_rows
                    if row["region"] == region
                    and row["split"] == "test"
                )

                ground_truth_path = Path(
                    region_row["label_path"]
                )

                if not ground_truth_path.is_absolute():
                    ground_truth_path = (
                        ROOT / ground_truth_path
                    )

                paths = [
                    otsu_preview_path,
                    otsu_prediction_path,
                    cva_preview_path,
                    cva_prediction_path,
                    ground_truth_path,
                ]

                for path in paths:
                    assert path.is_file(), path

                otsu_preview = np.asarray(
                    Image.open(
                        otsu_preview_path
                    ).convert("L")
                )
                otsu_prediction = np.asarray(
                    Image.open(
                        otsu_prediction_path
                    ).convert("L")
                )
                cva_preview = np.asarray(
                    Image.open(
                        cva_preview_path
                    ).convert("L")
                )
                cva_prediction = np.asarray(
                    Image.open(
                        cva_prediction_path
                    ).convert("L")
                )
                ground_truth = (
                    np.asarray(
                        Image.open(
                            ground_truth_path
                        ).convert("L")
                    )
                    > 0
                )

                images = [
                    otsu_preview,
                    otsu_prediction,
                    cva_preview,
                    cva_prediction,
                    ground_truth,
                ]
                titles = [
                    "Otsu difference",
                    "Otsu prediction",
                    "CVA magnitude",
                    "CVA prediction",
                    "Ground truth",
                ]

                figure, axes = plt.subplots(
                    1,
                    5,
                    figsize=(18, 4),
                )

                for axis, image, title in zip(
                    axes,
                    images,
                    titles,
                ):
                    axis.imshow(
                        image,
                        cmap="gray",
                    )
                    axis.set_title(title)
                    axis.axis("off")

                figure.suptitle(
                    f"OSCD test region: {{region}}"
                )
                figure.tight_layout()
                plt.show()
                """
            ).strip()
        ),
        new_markdown_cell(
            dedent(
                f"""
                ## Interpretation

                **Band Difference + Otsu** is the stronger classical
                baseline.

                - Test F1: **{otsu_test['f1_score']:.6f}**
                - Test IoU: **{otsu_test['iou']:.6f}**
                - Relative F1 improvement over CVA: **{f1_relative_improvement:.2%}**
                - Relative IoU improvement over CVA: **{iou_relative_improvement:.2%}**

                CVA + PCA + K-Means obtained higher recall
                ({cva_test['recall']:.6f}) but much lower precision
                ({cva_test['precision']:.6f}). It frequently grouped
                radiometric or seasonal variation with true land-cover
                change.

                Neither baseline learns spatial context, object boundaries
                or semantic land-use patterns. These limitations justify
                moving to a Siamese segmentation architecture.
                """
            ).strip()
        ),
        new_markdown_cell(
            dedent(
                f"""
                ## Week 3 target

                The Siamese U-Net must outperform the strongest frozen
                classical baseline:

                - **Target F1 greater than {strongest_metrics['f1_score']:.6f}**
                - **Target IoU greater than {strongest_metrics['iou']:.6f}**

                It should also produce more spatially coherent masks with
                fewer radiometric and seasonal false positives.
                """
            ).strip()
        ),
        new_code_cell(
            dedent(
                f"""
                assert eda["region_counts"] == {{
                    "train": 14,
                    "test": 10,
                    "overall": 24,
                }}

                assert (
                    eda[
                        "custom_hyderabad_aoi_used_for_metrics"
                    ]
                    is False
                )

                assert (
                    otsu["test_metrics"]["f1_score"]
                    == {otsu_test['f1_score']!r}
                )

                assert (
                    cva["test_metrics"]["f1_score"]
                    == {cva_test['f1_score']!r}
                )

                assert (
                    otsu["test_metrics"]["f1_score"]
                    > cva["test_metrics"]["f1_score"]
                )

                print("Week 2 notebook validation passed")
                print(
                    "Strongest baseline:",
                    "{strongest_name}",
                )
                print(
                    "Week 3 target F1:",
                    {strongest_metrics['f1_score']!r},
                )
                print(
                    "Week 3 target IoU:",
                    {strongest_metrics['iou']!r},
                )
                """
            ).strip()
        ),
    ]

    notebook = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
            "geowatch": {
                "week": 2,
                "dataset": "OSCD",
                "quantitative_scope": (
                    "OSCD labelled regions only"
                ),
                "hyderabad_used_for_metrics": False,
                "strongest_baseline": strongest_name,
                "representative_region": (
                    representative_region
                ),
            },
        },
    )

    return notebook


def write_notebook_atomic(
    notebook: NotebookNode,
    output_path: Path,
) -> None:
    """Write a notebook atomically."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_name(
        f"{output_path.stem}.tmp.ipynb"
    )

    nbformat.write(
        notebook,
        temporary_path,
    )

    temporary_path.replace(
        output_path
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the notebook-builder CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate the GeoWatch Week 2 EDA and classical "
            "baseline notebook from validated artifacts."
        )
    )

    parser.add_argument(
        "--eda-report",
        type=Path,
        default=Path(
            "reports/week2/eda/"
            "oscd_dataset_statistics.json"
        ),
    )
    parser.add_argument(
        "--otsu-report",
        type=Path,
        default=Path(
            "reports/week2/baselines/"
            "band_diff_otsu/"
            "band_diff_otsu_report.json"
        ),
    )
    parser.add_argument(
        "--cva-report",
        type=Path,
        default=Path(
            "reports/week2/baselines/"
            "cva_pca_kmeans/"
            "cva_pca_kmeans_report.json"
        ),
    )
    parser.add_argument(
        "--otsu-region-csv",
        type=Path,
        default=Path(
            "reports/week2/baselines/"
            "band_diff_otsu/"
            "oscd_region_metrics.csv"
        ),
    )
    parser.add_argument(
        "--cva-region-csv",
        type=Path,
        default=Path(
            "reports/week2/baselines/"
            "cva_pca_kmeans/"
            "oscd_region_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "notebooks/01_eda_baseline.ipynb"
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
    """Build the final Week 2 notebook."""
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
        eda = load_json(
            args.eda_report
        )
        otsu = load_json(
            args.otsu_report
        )
        cva = load_json(
            args.cva_report
        )

        otsu_rows = load_csv(
            args.otsu_region_csv
        )
        cva_rows = load_csv(
            args.cva_region_csv
        )

        validate_artifacts(
            eda=eda,
            otsu=otsu,
            cva=cva,
            otsu_rows=otsu_rows,
            cva_rows=cva_rows,
        )

        representative_region = (
            select_representative_region(
                otsu_rows
            )
        )

        notebook = build_notebook(
            eda=eda,
            otsu=otsu,
            cva=cva,
            representative_region=(
                representative_region
            ),
        )

        write_notebook_atomic(
            notebook=notebook,
            output_path=args.output,
        )

        print("Week 2 notebook generated")
        print("  Status: success")
        print(
            "  Representative region:",
            representative_region,
        )
        print(
            "  Cells:",
            len(notebook.cells),
        )
        print(
            "  Output:",
            args.output,
        )

        return 0

    except (
        FileNotFoundError,
        PermissionError,
        NotebookBuildError,
        ValueError,
        TypeError,
        OSError,
        csv.Error,
        json.JSONDecodeError,
        nbformat.ValidationError,
    ) as error:
        LOGGER.error(
            "%s",
            error,
        )
        return 1

    except Exception:
        LOGGER.exception(
            "Unexpected notebook-generation failure."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
