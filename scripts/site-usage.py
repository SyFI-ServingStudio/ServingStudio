"""Who used https://servingstudio.cs.washington.edu/, per UTC day, from the two
logs the site keeps:

- page views: the Intro site's visit.php log on the CSE web area
  (`<CSE_DIR>/logs/visits-<year>-<month>.jsonl`), read over ssh to CSE_HOST;
- public API use: the service's access log in the API host's live clone
  (`<PUBLIC_API_REMOTE_DIR>/ServingStudioSim/logs/public_api/access/<date>.jsonl`),
  read over ssh to PUBLIC_API_HOST.

    python3 scripts/site-usage.py [--days N]

Visitors are distinct (client address, user agent) pairs. Crawlers, our own
checks (curl, Python) and requests from the API host itself are counted apart,
not as visitors. The page-view log starts with the Intro deploy that added it;
the API log starts 2026-10-06.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta

BOT = re.compile(r"bot|crawl|spider|slurp|preview|headless", re.I)
TOOL = re.compile(r"^(curl|python|wget|go-http|node)", re.I)
API_PREFIX = "/api/public/v1"
TOP = 6


def ssh_cat(host, pattern):
    """Concatenated JSON lines of the files matching a remote glob."""
    command = f"cat {pattern} 2>/dev/null || true"
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, command],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def kind(row, own=()):
    ua = row.get("ua") or ""
    if row.get("client") in own or TOOL.match(ua):
        return "ours"
    if BOT.search(ua):
        return "bot"
    return "visitor"


def by_day(rows, since):
    days = defaultdict(list)
    for row in rows:
        day = row["ts"][:10]
        if day >= since:
            days[day].append(row)
    return days


def visitors(rows):
    return len({(r.get("client"), r.get("ua")) for r in rows})


def top(counter):
    return ", ".join(f"{k} {n}" for k, n in counter.most_common(TOP)) or "-"


def page_report(rows, since):
    print("== page views (visit.php)")
    if not rows:
        print("no records yet")
        return
    for day, day_rows in sorted(by_day(rows, since).items()):
        people = [r for r in day_rows if kind(r) == "visitor"]
        bots = len(day_rows) - len(people)
        pages = Counter(r.get("page", "").split("?")[0] for r in people)
        referrers = Counter(r["referrer"] for r in people if r.get("referrer"))
        print(f"{day}  views {len(people):4d}  visitors {visitors(people):3d}  other {bots}")
        print(f"            pages: {top(pages)}")
        if referrers:
            print(f"            from:  {top(referrers)}")


def api_report(rows, since, own):
    print("\n== public API (access log)")
    if not rows:
        print("no records")
        return
    for day, day_rows in sorted(by_day(rows, since).items()):
        people = [r for r in day_rows if kind(r, own) == "visitor"]
        others = len(day_rows) - len(people)
        work = Counter(
            r["path"].removeprefix(API_PREFIX)
            for r in people
            if r["method"] == "POST"
        )
        print(
            f"{day}  requests {len(people):4d}  visitors {visitors(people):3d}  "
            f"other {others}  runs: {top(work)}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--days", type=int, default=7, help="UTC days to show, default 7")
    args = parser.parse_args()

    cse_host = os.environ.get("CSE_HOST", "bicycle")
    cse_dir = os.environ.get("CSE_DIR", "/cse/web/research/servingstudio")
    api_host = os.environ.get("PUBLIC_API_HOST")
    api_bind = os.environ.get("PUBLIC_API_BIND")
    remote_dir = os.environ.get("PUBLIC_API_REMOTE_DIR", "ServingStudio")

    today = datetime.now(UTC).date()
    since = (today - timedelta(days=args.days - 1)).isoformat()
    months = sorted({(today - timedelta(days=d)).strftime("%Y-%m") for d in range(args.days)})

    page_rows = []
    for month in months:
        page_rows += ssh_cat(cse_host, f"{cse_dir}/logs/visits-{month}.jsonl")
    page_report(page_rows, since)

    if not api_host:
        print("\nset PUBLIC_API_HOST to include the public API's log", file=sys.stderr)
        return
    days = " ".join(
        f"{remote_dir}/ServingStudioSim/logs/public_api/access/"
        f"{(today - timedelta(days=d)).isoformat()}.jsonl"
        for d in range(args.days)
    )
    # The API host's own checks reach the service from these addresses: its
    # Docker bridge, its bind address, and (through the site) its public one.
    own = {"172.17.0.1", "127.0.0.1", api_bind}
    api_report(ssh_cat(api_host, days), since, own)


if __name__ == "__main__":
    main()
