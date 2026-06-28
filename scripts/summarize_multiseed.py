from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


FINAL_RE = re.compile(
    r"epoch\s+(?P<epoch>\d+)\s+done.*?"
    r"val_top1=(?P<val_top1>\d+\.\d+)\s+"
    r"val_top5=(?P<val_top5>\d+\.\d+)\s+"
    r"test_top1=(?P<test_top1>\d+\.\d+)\s+"
    r"test_top5=(?P<test_top5>\d+\.\d+)"
)


def parse_log(path: Path) -> dict[str, float] | None:
    last_match = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = FINAL_RE.search(line)
        if match:
            last_match = match
    if last_match is None:
        return None
    values = {key: float(value) for key, value in last_match.groupdict().items()}
    values["test_t1_err"] = 1.0 - values["test_top1"]
    values["test_t5_err"] = 1.0 - values["test_top5"]
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("./outputs/multiseed/logs"))
    args = parser.parse_args()

    rows = []
    for path in sorted(args.log_dir.glob("*.log")):
        result = parse_log(path)
        if result is None:
            print(f"missing-final,{path.name}")
            continue
        parts = path.stem.split("_")
        seed = parts[-1].replace("seed", "")
        method = parts[-2]
        dataset = parts[0]
        rows.append((dataset, method, seed, result))
        print(
            f"{dataset},{method},seed={seed},"
            f"top1={result['test_top1']:.4f},"
            f"t1_err={result['test_t1_err'] * 100:.2f},"
            f"top5={result['test_top5']:.4f},"
            f"t5_err={result['test_t5_err'] * 100:.2f}"
        )

    print("\nsummary")
    groups: dict[tuple[str, str], list[dict[str, float]]] = {}
    for dataset, method, _, result in rows:
        groups.setdefault((dataset, method), []).append(result)
    for (dataset, method), results in sorted(groups.items()):
        t1 = np.asarray([item["test_t1_err"] * 100.0 for item in results])
        t5 = np.asarray([item["test_t5_err"] * 100.0 for item in results])
        print(
            f"{dataset},{method},n={len(results)},"
            f"t1_err_mean={t1.mean():.2f},t1_err_std={t1.std(ddof=0):.2f},"
            f"t5_err_mean={t5.mean():.2f},t5_err_std={t5.std(ddof=0):.2f}"
        )


if __name__ == "__main__":
    main()
