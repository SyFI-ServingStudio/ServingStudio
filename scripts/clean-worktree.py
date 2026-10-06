"""Remove the regenerable bulk from a worktree before retiring it to old-wt/.

    python3 scripts/clean-worktree.py WORKTREE... [--apply]

Without --apply it only reports what would go. It removes:

- Cargo build output: a `target/` beside a `Cargo.toml` (marked by CACHEDIR.TAG
  or .rustc_info.json, including rust-analyzer flycheck dirs). Rebuild.
- Under any `logs/` directory:
  - Simulator Perfetto traces, `*.pftrace` and `*.pftrace.gz`. Rerun the simulation.
  - Plots, `*.png`. Rerun the Analyzer render.
  - Analyzer payload shards, `payloads/*.jsonl` and `payloads/*.jsonl.zst`.
    Rerun the Analyzer; it reads the parsed capture (`parsed.json[.zst]`,
    `host_timeline.json`), which is kept.
  - nsys SQLite exports, `X.sqlite` only when `X.nsys-rep` sits beside it.
    Re-export with `nsys export --type sqlite`.

It keeps everything that needs a GPU or a past state to recreate: `.nsys-rep`
captures, parsed captures and other JSON, parquet, presets, source, `profile.db`
and the goal/progress/notes records. Analyzer configs hold absolute paths, so a
regeneration after the move must point them at the old-wt/ location.

With --apply, each worktree gets `.worktree-clean.tsv`: one row per category and
directory with the removed file count and bytes, so later readers know what to
regenerate.
"""

import argparse
import os
import shutil
import sys
from collections import defaultdict
from datetime import UTC, datetime

SKIP_DIRS = {".git", ".venv", "node_modules"}
CATEGORIES = {
    "cargo-target": "Cargo target/ (rebuild)",
    "sim-trace": "logs: simulator .pftrace(.gz) (rerun simulation)",
    "plot": "logs: .png plots (rerun Analyzer render)",
    "analyzer-payload": "logs: Analyzer payloads/*.jsonl(.zst) (rerun Analyzer)",
    "nsys-sqlite": "logs: .sqlite beside its .nsys-rep (nsys export)",
}
MANIFEST = ".worktree-clean.tsv"


def disk_bytes(st):
    return st.st_blocks * 512


def tree_bytes(path):
    total = files = 0
    for root, _dirs, names in os.walk(path):
        for name in names:
            try:
                st = os.lstat(os.path.join(root, name))
            except OSError:
                continue
            total += disk_bytes(st)
            files += 1
    return files, total


def is_cargo_target(path):
    return (
        os.path.isfile(os.path.join(os.path.dirname(path), "Cargo.toml"))
        and any(os.path.exists(os.path.join(path, m)) for m in ("CACHEDIR.TAG", ".rustc_info.json"))
    )


def classify(root, name, names, in_logs):
    """Category of a file under logs/, or None to keep it."""
    if not in_logs:
        return None
    if name.endswith((".pftrace", ".pftrace.gz")):
        return "sim-trace"
    if name.endswith(".png"):
        return "plot"
    if os.path.basename(root) == "payloads" and name.endswith((".jsonl", ".jsonl.zst")):
        return "analyzer-payload"
    if name.endswith(".sqlite") and name[: -len(".sqlite")] + ".nsys-rep" in names:
        return "nsys-sqlite"
    return None


def scan(worktree):
    """Yield (category, path, is_dir, files, bytes) for everything to remove."""
    for root, dirs, names in os.walk(worktree):
        in_logs = "logs" in os.path.relpath(root, worktree).split(os.sep)
        keep = []
        for d in dirs:
            path = os.path.join(root, d)
            if d in SKIP_DIRS or os.path.islink(path):
                continue
            if d == "target" and is_cargo_target(path):
                yield ("cargo-target", path, True, *tree_bytes(path))
                continue
            keep.append(d)
        dirs[:] = keep
        name_set = set(names)
        for name in names:
            category = classify(root, name, name_set, in_logs)
            if category is None:
                continue
            path = os.path.join(root, name)
            try:
                st = os.lstat(path)
            except OSError:
                continue
            yield (category, path, False, 1, disk_bytes(st))


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1000


def clean(worktree, apply):
    by_category = defaultdict(lambda: [0, 0])
    by_dir = defaultdict(lambda: [0, 0])
    errors = []
    for category, path, is_dir, files, size in scan(worktree):
        if apply:
            try:
                shutil.rmtree(path) if is_dir else os.remove(path)
            except OSError as exc:
                errors.append(f"{path}: {exc}")
                continue
        by_category[category][0] += files
        by_category[category][1] += size
        where = path if is_dir else os.path.dirname(path)
        by_dir[(category, os.path.relpath(where, worktree))][0] += files
        by_dir[(category, os.path.relpath(where, worktree))][1] += size

    verb = "removed" if apply else "would remove"
    total = sum(b for _f, b in by_category.values())
    print(f"{worktree}: {verb} {human(total)}")
    for category, label in CATEGORIES.items():
        files, size = by_category.get(category, (0, 0))
        if files:
            print(f"  {human(size):>9}  {files:>9} files  {label}")
    for err in errors[:20]:
        print(f"  error: {err}", file=sys.stderr)
    if apply and by_dir:
        stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(os.path.join(worktree, MANIFEST), "a") as fh:
            for (category, where), (files, size) in sorted(by_dir.items()):
                fh.write(f"{stamp}\t{category}\t{where}\t{files}\t{size}\n")
    return total, len(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("worktrees", nargs="+", help="worktree directories to clean")
    parser.add_argument("--apply", action="store_true", help="delete; without it, only report")
    args = parser.parse_args()

    grand = failures = 0
    for worktree in args.worktrees:
        worktree = os.path.abspath(worktree)
        if not os.path.isdir(worktree):
            parser.error(f"not a directory: {worktree}")
        if not os.path.exists(os.path.join(worktree, ".git")):
            print(f"{worktree}: no .git, not a worktree; skipped", file=sys.stderr)
            continue
        total, errs = clean(worktree, args.apply)
        grand += total
        failures += errs
    if len(args.worktrees) > 1:
        print(f"total: {'removed' if args.apply else 'would remove'} {human(grand)}")
    if not args.apply:
        print("dry run; pass --apply to delete")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
