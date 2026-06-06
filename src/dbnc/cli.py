"""Command-line interface for the DBNC experiment pipeline.

Exposes the ``dbnc`` console script with subcommands for validation, the
benchmark, and report generation, e.g.::

    dbnc benchmark --methods dbnc xgboost --datasets car vote --seeds 0 1 2 3 4
    dbnc report --output reports

Run ``dbnc --help`` for the full list of subcommands and options.
"""
from dbnc._pipeline import build_parser, main  # noqa: F401

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main(None))
