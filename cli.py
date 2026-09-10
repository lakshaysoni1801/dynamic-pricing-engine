"""Command-line interface.

    pricing-engine eval     # train, evaluate, register+promote a model
    pricing-engine serve    # run the FastAPI app with uvicorn
"""
from __future__ import annotations

import argparse


def _eval(args) -> int:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
    import run_eval

    return run_eval.main(["--json", args.json] if args.json else [])


def _serve(args) -> int:
    import uvicorn

    uvicorn.run("pricing_engine.service:app", host=args.host, port=args.port, reload=False)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="pricing-engine")
    sub = parser.add_subparsers(dest="command", required=True)

    pe = sub.add_parser("eval")
    pe.add_argument("--json")

    ps = sub.add_parser("serve")
    ps.add_argument("--host", default="0.0.0.0")
    ps.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    if args.command == "eval":
        return _eval(args)
    if args.command == "serve":
        return _serve(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
