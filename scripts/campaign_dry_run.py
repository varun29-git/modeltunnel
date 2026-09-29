"""Print the cost-capped campaign plan (dry run only — launches nothing).

Usage: .venv/bin/python scripts/campaign_dry_run.py [--cap-hours 60] [--output docs/campaign_dry_run.md]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from terminus.campaign import Campaign, dry_run_report
from terminus.registry import Ledger
from terminus.search_plan import campaign_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap-hours", type=float, default=60.0)
    parser.add_argument("--output")
    args = parser.parse_args()
    with __import__("tempfile").TemporaryDirectory() as root:
        campaign = Campaign(args.cap_hours, Ledger(Path(root) / "ledger.jsonl"))
        report = dry_run_report(campaign_plan(), campaign=campaign)
    print(report["text"])
    if args.output:
        Path(args.output).write_text("# Dry-run campaign plan (no jobs launched)\n\n```\n"
                                     + report["text"] + "\n```\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
