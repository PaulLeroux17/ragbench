from __future__ import annotations
import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Default directory for scripts
DEFAULT_DIR = "/app/scripts"

# Hard-coded alias mapping
ALIASES: Dict[str, str] = {
    "bm25": "run_bm25_grid.py",
    "bi_encoder": "run_bi_encoder.py",
}


def scripts_dir() -> Path:
    """Return the base directory for scripts (can be overridden via env var)."""
    return Path(os.environ.get("RAGBENCH_SCRIPTS_DIR", DEFAULT_DIR))


def resolve_alias(alias: str, sdir: Optional[str]) -> Optional[Path]:
    """Resolve alias to absolute path if known."""
    script_name = ALIASES.get(alias)
    if not script_name:
        return None
    base = Path(sdir) if sdir else scripts_dir()
    return (base / script_name).resolve()


def list_available(sdir: Optional[str]) -> List[Tuple[str, str]]:
    """Return [(alias, filename)] for all aliases."""
    rows: List[Tuple[str, str]] = []
    base = Path(sdir) if sdir else scripts_dir()
    for alias, script in sorted(ALIASES.items()):
        p = base / script
        status = "OK" if p.exists() else "MISSING"
        rows.append((alias, f"{p.name} [{status}]"))
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Parse known args so everything after the alias is passed through.
    parser = argparse.ArgumentParser(
        prog="ragbench",
        description="Dispatcher for executing fixed script aliases.",
        add_help=True,
    )
    parser.add_argument(
        "--list", action="store_true", help="List available commands and exit"
    )
    parser.add_argument(
        "--scripts-dir",
        default=None,
        help=f"Scripts directory (default: {DEFAULT_DIR})",
    )
    parser.add_argument(
        "alias",
        nargs="?",
        choices=sorted(ALIASES.keys()),
        help="Script alias to run",
    )
    args, passthru = parser.parse_known_args(argv)

    if args.scripts_dir:
        os.environ["RAGBENCH_SCRIPTS_DIR"] = args.scripts_dir

    # --list or no alias → print available
    if args.list or not args.alias:
        print("Available commands:")
        for name, desc in list_available(args.scripts_dir):
            print(f"  - {name:12s} {desc}")
        if not args.list and not args.alias:
            print("\nUsage examples:")
            print("  ragbench bm25 --config configs/scidocs_bm25.yaml")
            print(
                "  ragbench build_index --config configs/build_index_scidocs.yaml"
            )
        return 0

    # Resolve alias → script
    target = resolve_alias(args.alias, args.scripts_dir)
    if target is None:
        print(f"Error: unknown command '{args.alias}'. Try: ragbench --list")
        return 1
    if not target.exists():
        print(f"Error: script not found: {target}")
        return 1

    # Run the script and pass through remaining args (e.g., --config ...)
    cmd = [sys.executable, str(target), *passthru]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
