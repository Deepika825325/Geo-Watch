"""Create and audit the controlled GeoWatch plain-BCE ablation."""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


EXPECTED_VALIDATION_REGIONS = frozenset(
    {
        "hongkong",
        "mumbai",
        "paris",
    }
)

PATH_FIELDS_ALLOWED_TO_CHANGE = (
    "checkpoint_directory",
    "log_directory",
    "prediction_directory",
)


def require_mapping(
    value: Any,
    name: str,
) -> Mapping[str, Any]:
    """Return a validated configuration mapping."""

    if not isinstance(
        value,
        Mapping,
    ):
        raise ValueError(
            f"{name} must be a mapping."
        )

    return value


def load_yaml_config(
    path: Path,
) -> dict[str, Any]:
    """Load one YAML configuration as a mutable dictionary."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as input_file:
        loaded = yaml.safe_load(
            input_file
        )

    return dict(
        require_mapping(
            loaded,
            "root configuration",
        )
    )


def normalize_for_fairness_check(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove fields that are allowed to differ in the ablation."""

    normalized = copy.deepcopy(
        dict(
            config
        )
    )

    project = dict(
        require_mapping(
            normalized.get(
                "project"
            ),
            "project",
        )
    )
    project.pop(
        "experiment_name",
        None,
    )
    normalized[
        "project"
    ] = project

    paths = dict(
        require_mapping(
            normalized.get(
                "paths"
            ),
            "paths",
        )
    )

    for field_name in PATH_FIELDS_ALLOWED_TO_CHANGE:
        paths.pop(
            field_name,
            None,
        )

    normalized[
        "paths"
    ] = paths
    normalized.pop(
        "loss",
        None,
    )

    return normalized


def validate_source_protocol(
    config: Mapping[str, Any],
) -> None:
    """Validate the source experiment and sealed evaluation protocol."""

    protocol = require_mapping(
        config.get(
            "protocol"
        ),
        "protocol",
    )
    dataset = require_mapping(
        config.get(
            "dataset"
        ),
        "dataset",
    )
    metrics = require_mapping(
        config.get(
            "metrics"
        ),
        "metrics",
    )
    loss = require_mapping(
        config.get(
            "loss"
        ),
        "loss",
    )

    if protocol.get(
        "official_test_regions_sealed"
    ) is not True:
        raise ValueError(
            "Official test regions must remain sealed."
        )

    validation_regions = frozenset(
        str(region).strip().lower()
        for region in dataset[
            "validation_regions"
        ]
    )

    if validation_regions != EXPECTED_VALIDATION_REGIONS:
        raise ValueError(
            "The BCE ablation must use Hong Kong, Mumbai and Paris."
        )

    training_regions = {
        str(region).strip().lower()
        for region in dataset[
            "train_regions"
        ]
    }

    if training_regions.intersection(
        validation_regions
    ):
        raise ValueError(
            "Training and validation regions overlap."
        )

    if str(
        loss.get(
            "name",
            ""
        )
    ).strip().lower() != "dice_focal":
        raise ValueError(
            "The source experiment must use Dice+Focal."
        )

    if float(
        metrics[
            "threshold"
        ]
    ) != 0.5:
        raise ValueError(
            "The controlled training comparison must retain threshold 0.5."
        )


def build_bce_ablation_config(
    source_config: Mapping[str, Any],
    experiment_root: Path,
) -> dict[str, Any]:
    """Create a BCE config that differs only in allowed fields."""

    validate_source_protocol(
        source_config
    )

    ablation_config = copy.deepcopy(
        dict(
            source_config
        )
    )

    project = dict(
        require_mapping(
            ablation_config.get(
                "project"
            ),
            "project",
        )
    )
    project[
        "experiment_name"
    ] = "week5_ablation_bce"
    ablation_config[
        "project"
    ] = project

    paths = dict(
        require_mapping(
            ablation_config.get(
                "paths"
            ),
            "paths",
        )
    )
    paths[
        "checkpoint_directory"
    ] = str(
        experiment_root
        / "checkpoints"
    )
    paths[
        "log_directory"
    ] = str(
        experiment_root
        / "logs"
    )
    paths[
        "prediction_directory"
    ] = str(
        experiment_root
        / "predictions"
    )
    ablation_config[
        "paths"
    ] = paths

    ablation_config[
        "loss"
    ] = {
        "name": "bce",
        "reduction": "mean",
    }

    if (
        normalize_for_fairness_check(
            source_config
        )
        != normalize_for_fairness_check(
            ablation_config
        )
    ):
        raise ValueError(
            "The BCE configuration changed fields outside "
            "the controlled ablation contract."
        )

    return ablation_config


def write_yaml_config(
    path: Path,
    config: Mapping[str, Any],
) -> None:
    """Write a deterministic YAML experiment configuration."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        yaml.safe_dump(
            dict(
                config
            ),
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def write_contract(
    path: Path,
    source_config_path: Path,
    output_config_path: Path,
    config: Mapping[str, Any],
) -> None:
    """Write a machine-readable controlled-ablation contract."""

    dataset = require_mapping(
        config.get(
            "dataset"
        ),
        "dataset",
    )
    loader = require_mapping(
        dataset.get(
            "loader"
        ),
        "dataset.loader",
    )
    training = require_mapping(
        config.get(
            "training"
        ),
        "training",
    )
    optimizer = require_mapping(
        config.get(
            "optimizer"
        ),
        "optimizer",
    )
    metrics = require_mapping(
        config.get(
            "metrics"
        ),
        "metrics",
    )

    payload = {
        "experiment": "week5_ablation_bce",
        "source_config": str(
            source_config_path
        ),
        "output_config": str(
            output_config_path
        ),
        "controlled_change": {
            "from": "dice_focal",
            "to": "plain_bce",
        },
        "frozen_fields": {
            "train_regions": list(
                dataset[
                    "train_regions"
                ]
            ),
            "validation_regions": list(
                dataset[
                    "validation_regions"
                ]
            ),
            "batch_size": int(
                loader[
                    "batch_size"
                ]
            ),
            "epochs": int(
                training[
                    "epochs"
                ]
            ),
            "learning_rate": float(
                optimizer[
                    "learning_rate"
                ]
            ),
            "training_metric_threshold": float(
                metrics[
                    "threshold"
                ]
            ),
        },
        "protocol": {
            "official_test_regions_accessed": False,
            "official_test_labels_accessed": False,
        },
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the BCE ablation configuration command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Create a controlled plain-BCE GeoWatch ablation "
            "from the frozen Dice+Focal configuration."
        )
    )

    parser.add_argument(
        "--source-config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--contract-output",
        type=Path,
        required=True,
    )

    return parser


def main() -> None:
    """Create and report the controlled BCE experiment configuration."""

    arguments = build_argument_parser().parse_args()

    source_config = load_yaml_config(
        arguments.source_config
    )

    ablation_config = build_bce_ablation_config(
        source_config=source_config,
        experiment_root=arguments.experiment_root,
    )

    write_yaml_config(
        path=arguments.output_config,
        config=ablation_config,
    )

    write_contract(
        path=arguments.contract_output,
        source_config_path=arguments.source_config,
        output_config_path=arguments.output_config,
        config=ablation_config,
    )

    dataset = require_mapping(
        ablation_config[
            "dataset"
        ],
        "dataset",
    )
    loader = require_mapping(
        dataset[
            "loader"
        ],
        "dataset.loader",
    )
    training = require_mapping(
        ablation_config[
            "training"
        ],
        "training",
    )

    print(
        "GeoWatch BCE ablation configuration created"
    )
    print(
        "  Experiment:",
        ablation_config[
            "project"
        ][
            "experiment_name"
        ],
    )
    print(
        "  Loss:",
        ablation_config[
            "loss"
        ][
            "name"
        ],
    )
    print(
        "  Train regions:",
        len(
            dataset[
                "train_regions"
            ]
        ),
    )
    print(
        "  Validation regions:",
        len(
            dataset[
                "validation_regions"
            ]
        ),
    )
    print(
        "  Batch size:",
        loader[
            "batch_size"
        ],
    )
    print(
        "  Epoch limit:",
        training[
            "epochs"
        ],
    )
    print(
        "  Training metric threshold:",
        ablation_config[
            "metrics"
        ][
            "threshold"
        ],
    )
    print(
        "  Output config:",
        arguments.output_config,
    )
    print(
        "  Contract:",
        arguments.contract_output,
    )
    print(
        "  Official test regions accessed:",
        False,
    )
    print(
        "  Official test labels accessed:",
        False,
    )


if __name__ == "__main__":
    main()
