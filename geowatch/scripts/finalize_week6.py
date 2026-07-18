from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main(
) -> int:
    project_root = Path(
        "."
    ).resolve()

    command = [
        sys.executable,
        "-m",
        "src.evaluation.week6_report",
        "--project-root",
        str(
            project_root
        ),
        "--run-tests",
        "--python",
        sys.executable,
    ]

    completed = subprocess.run(
        command,
        cwd=project_root,
        check=False,
    )

    return int(
        completed.returncode
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
