from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bolr.data.feature_manifest import render_feature_audit_markdown, write_feature_manifest
from bolr.data.leakage_audit import audit_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/YM_full.parquet")
    parser.add_argument("--manifest", default="data/manifests/feature_manifest.csv")
    parser.add_argument("--audit", default="data/manifests/feature_audit.md")
    args = parser.parse_args()

    rows, lag_results = audit_features(args.dataset)
    frame = write_feature_manifest(rows, args.manifest)
    render_feature_audit_markdown(frame, args.audit)
    print(f"wrote_manifest={args.manifest}")
    print(f"wrote_audit={args.audit}")
    for result in lag_results:
        print(
            f"lag_check column={result.column_name} verified={result.verified} "
            f"max_abs_error={result.max_abs_error} note={result.note}"
        )


if __name__ == "__main__":
    main()
