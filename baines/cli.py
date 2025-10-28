"""
Command-line interface for the Baines compressor sizing tool.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from math import isfinite
from typing import Iterable, Optional

from .core import Params, ResultRow, SizingInputs, design_impeller


def _fmt(x: Optional[float], wid: int = 9, prec: int = 5) -> str:
    if x is None:
        return " " * wid
    if isinstance(x, float) and not isfinite(x):
        return " " * (wid - 3) + "nan"
    return f"{x:>{wid}.{prec}f}"


def print_table(rows: Iterable[ResultRow], include_dims: bool = False) -> None:
    head = (
        "beta_b2  b2/r2   U2/A0    M2      M2_rel  M1      M1_rel  M1r/M2r  W1/W2   Ns(stag)  Ns(static)  THETA    PHI      PSI      WND"
    )
    print(head)
    print("-" * len(head))
    for r in rows:
        print(
            f"{r.BETAB2:>7.2f}"
            f"{_fmt(r.B2R2)}{_fmt(r.U2A01)}{_fmt(r.AM2)}{_fmt(r.AM2R)}{_fmt(r.AM1)}{_fmt(r.AM1R)}"
            f"{_fmt(r.AMR)}{_fmt(r.WR)}{_fmt(r.SS)}{_fmt(r.SSG)}{_fmt(r.THETA)}{_fmt(r.PHI)}{_fmt(r.PSI)}{_fmt(r.WND)}"
        )
        if include_dims and r.r2_mm is not None:
            print(
                f"    -> r2={r.r2_mm:.1f} mm,  b2={r.b2_mm:.1f} mm,  "
                f"r1s={r.r1s_mm:.1f} mm,  r1h={r.r1h_mm:.1f} mm"
            )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Baines BASIC compressor impeller design (Python translation)"
    )
    ap.add_argument("--pr", type=float, required=True, help="Design pressure ratio (stage Po2/Po1)")
    ap.add_argument(
        "--beta2-start",
        type=float,
        required=True,
        help="Starting impeller discharge blade metal angle, deg",
    )
    ap.add_argument(
        "--beta2-step",
        type=float,
        required=True,
        help="Step length for blade angle, deg (positive)",
    )
    ap.add_argument("--nsteps", type=int, required=True, help="Number of steps")
    ap.add_argument("--sf", type=float, default=Params.SF, help="Slip factor")
    ap.add_argument("--eti", type=float, default=Params.ETI, help="Impeller total-to-total efficiency")
    ap.add_argument("--ets", type=float, default=Params.ETS, help="Stage total-to-total efficiency")
    ap.add_argument(
        "--beta1s",
        type=float,
        default=Params.BETA1S,
        help="Inlet shroud relative flow angle (deg)",
    )
    ap.add_argument(
        "--anu",
        type=float,
        default=Params.ANU,
        help="Inducer hub-to-shroud radius ratio r1h/r1s",
    )
    ap.add_argument(
        "--alp2",
        type=float,
        default=Params.ALP2,
        help="Discharge absolute flow angle (deg)",
    )
    ap.add_argument(
        "--r1sr2",
        type=float,
        default=Params.R1SR2,
        help="Inducer shroud to discharge tip radius ratio r1s/r2",
    )
    ap.add_argument("--g", type=float, default=Params.G, help="Ratio of specific heats gamma")
    ap.add_argument("--mf", type=float, help="Mass flow rate [kg/s] (optional)")
    ap.add_argument("--p01bar", type=float, help="Inlet stagnation pressure [bar] (optional)")
    ap.add_argument("--t01", type=float, help="Inlet stagnation temperature [K] (optional)")
    return ap


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    params = Params(
        SF=args.sf,
        ETI=args.eti,
        ETS=args.ets,
        BETA1S=args.beta1s,
        ANU=args.anu,
        ALP2=args.alp2,
        R1SR2=args.r1sr2,
        G=args.g,
    )
    sizing = SizingInputs(mf=args.mf, p01_bar=args.p01bar, t01=args.t01)

    rows = design_impeller(
        args.pr,
        args.beta2_start,
        args.beta2_step,
        args.nsteps,
        params=params,
        sizing=sizing,
    )
    include_dims = sizing.mf is not None and sizing.p01_bar is not None and sizing.t01 is not None

    print("\nDESIGN PRESSURE RATIO:", args.pr)
    print(
        "Parameters:",
        ", ".join(f"{k}={v}" for k, v in asdict(params).items() if k != "Rgas"),
        f"Rgas={params.Rgas}",
    )
    if include_dims:
        print(
            f"Sizing inputs: mf={sizing.mf} kg/s, p01={sizing.p01_bar} bar, t01={sizing.t01} K"
        )
    print()
    print_table(rows, include_dims=include_dims)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
