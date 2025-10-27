#!/usr/bin/env python3
"""
Baines-style preliminary design of a centrifugal compressor impeller (no inlet prewhirl).

This script re-implements the GWBASIC program listing from Appendix A ("Preliminary design
of centrifugal compressor impellers") in the uploaded PDF (Baines_Compressor.pdf).
It follows the same variable names and non-dimensional relations where practical.

Key differences vs. the original:
- Uses general gamma (ratio of specific heats), not the hard-coded .4 = (gamma-1).
- Provides a simple CLI and a reusable function API.
- Safer math (guards for domain errors), and clearer printed tables.

Coordinates/angles:
- ALP2 (alpha_2): absolute flow angle at impeller discharge, degrees.
- BETAB2 (beta_b2): blade metal angle at discharge (backsweep negative, radially back-swept blades give negative beta_b2).
- BETA1S (beta_1s): inlet shroud relative flow angle, degrees (often negative for compressors).
- All trig inputs are in degrees and converted to radians internally.

Assumptions:
- No inlet prewhirl (W_t1/A01 = -U1/A01).
- Ideal-gas, isentropic relations where used.
- Program matches the BASIC logic as closely as possible, including definitions of
  PHI (volume-flow coeff), THETA (mass-flow coeff), PSI (head coeff), and specific speeds.

Formulas mirror the BASIC lines as reconstructed from the PDF:
- AL = SF / (1 - tan(BETAB2)/tan(ALP2))
- U2/A01 and C_t2/A01 follow from pressure ratio and AL, ETS, etc.
- T0- and static conversions use standard compressible-flow relations.

Usage:
    python3 baines_compressor.py --pr 2.5 --beta2-start 0 --beta2-step 10 --nsteps 5
    # Optional sizing (adds geometric dimensions):
    python3 baines_compressor.py --pr 2.5 --beta2-start 0 --beta2-step 10 --nsteps 5 \
        --mf 0.3 --p01bar 1.0 --t01 288

Author: Re-implemented by ChatGPT (Python translation).
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from math import tan, cos, sin, atan, sqrt, pi, radians, degrees, isfinite

@dataclass
class Params:
    SF: float = 0.85       # slip factor
    ETI: float = 0.90      # impeller total-to-total efficiency
    ETS: float = 0.80      # stage  total-to-total efficiency
    BETA1S: float = -60.0  # inlet shroud relative flow angle [deg]
    ANU: float = 0.40      # r1h / r1s (hub/shroud radius ratio at inducer)
    ALP2: float = 65.0     # discharge absolute flow angle [deg]
    R1SR2: float = 0.70    # r1s / r2 (inducer shroud to discharge tip radius ratio)
    G: float = 1.40        # ratio of specific heats (gamma)
    Rgas: float = 287.0    # J/(kg*K), specific gas constant for air

@dataclass
class SizingInputs:
    mf: Optional[float] = None     # mass flow rate [kg/s]
    p01_bar: Optional[float] = None# inlet stagnation pressure [bar]
    t01: Optional[float] = None    # inlet stagnation temperature [K]

@dataclass
class ResultRow:
    BETAB2: float
    B2R2: float
    U2A01: float
    AM2: float
    AM2R: float
    AM1: float
    AM1R: float
    AMR: float
    WR: float
    SS: float
    SSG: float
    THETA: float
    PHI: float
    PSI: float
    WND: float
    # For optional geometric sizing (when mf, p01, t01 supplied)
    r2_mm: Optional[float] = None
    b2_mm: Optional[float] = None
    r1s_mm: Optional[float] = None
    r1h_mm: Optional[float] = None

def _safe_sqrt(x: float) -> float:
    return sqrt(x) if x > 0.0 else float('nan')

def _guard(x: float, eps: float = 1e-12) -> float:
    # Avoid division by zero and invalid tan/cos for angles near singularities
    if abs(x) < eps:
        return float('nan')
    return x

def design_step(pr: float, betab2_deg: float, P: Params, S: SizingInputs) -> ResultRow:
    # Angles (rad) and basic trig
    beta1 = radians(P.BETA1S)
    tb1s = tan(beta1)
    cb1s = cos(beta1)
    alp2r = radians(P.ALP2)
    sa2 = sin(alp2r)
    ca2 = cos(alp2r)
    ta2 = tan(alp2r)

    betb2 = radians(betab2_deg)
    tbb2 = tan(betb2)

    # AL from slip model and geometry
    denom = 1.0 - (tbb2 / _guard(ta2))
    AL = P.SF / denom if isfinite(denom) else float('nan')

    # U2/A01 from pressure ratio (general gamma form)
    pr_term = pr**((P.G - 1.0) / P.G) - 1.0
    denom_u = (P.G - 1.0) * P.ETS * AL
    U2A01 = _safe_sqrt(pr_term / denom_u) if isfinite(denom_u) else float('nan')

    # Tangential absolute at 2 (non-dim with A01)
    CT2A01 = U2A01 * AL

    # T0 ratio across rotor from Euler (ΔT0/T0 = (γ-1)*U2/A0 * Ct2/A0)
    TO2TO1 = 1.0 + (P.G - 1.0) * U2A01 * CT2A01

    # Impeller total pressure ratio using ETI
    PO2PO1 = (1.0 + P.ETI * (TO2TO1 - 1.0)) ** (P.G / (P.G - 1.0))

    # Absolute speed at 2, scaled to A01 then to a2
    C2A01 = CT2A01 / _guard(sa2)
    C2A02 = C2A01 / _safe_sqrt(TO2TO1)  # divide by sqrt(T0 ratio) to keep with A0 scaling
    T2TO2 = 1.0 - 0.5 * (P.G - 1.0) * (C2A02 ** 2)  # static-to-stagnation at 2
    TO2T2 = 1.0 / T2TO2 if isfinite(T2TO2) else float('nan')

    # Absolute Mach at 2
    AM2 = C2A02 * _safe_sqrt(TO2T2)

    # Convert U2/A01 to U2/a2 and Ct2/a2
    U2A02 = U2A01 / _safe_sqrt(TO2TO1)
    U2A2  = U2A02 * _safe_sqrt(T2TO2)
    CT2A2 = CT2A01 * _safe_sqrt(T2TO2 / TO2TO1) if isfinite(TO2TO1) else float('nan')

    # Tangential component of relative at 2 and meridional component
    WT2A2 = CT2A2 - U2A2
    CM2A2 = AM2 * ca2  # since AM2 = C2/a2, meridional Mach = M2 * cos(alpha2)

    # Relative Mach at 2
    AM2R = _safe_sqrt(WT2A2 ** 2 + CM2A2 ** 2)

    # Relative flow angle at 2 (deg): beta2 = atan(W_t / C_m)
    B2 = degrees(atan(WT2A2 / _guard(CM2A2)))

    # Stagnation-to-static & to upstream stagnation
    PO2P2 = (TO2T2) ** (P.G / (P.G - 1.0)) if isfinite(TO2T2) else float('nan')
    P2PO1 = PO2PO1 / _guard(PO2P2)

    # Density ratios
    R2RO1 = P2PO1 * (T2TO2 / _guard(TO2TO1))

    # Meridional absolute at 2 scaled by A01
    CM2A01 = CT2A01 / _guard(ta2)

    # Inlet wheel speed (shroud) non-dim by A01
    U1A01 = U2A01 * P.R1SR2

    # No inlet prewhirl: W_t1/A01 = -U1/A01 ; relate to absolute via beta1s
    WT1A01 = -U1A01
    C1A01 = WT1A01 / _guard(tb1s)

    # Absolute Mach at inlet 1 using T1/T01 relation
    T1T01 = 1.0 - 0.5 * (P.G - 1.0) * (C1A01 ** 2)
    AM1 = C1A01 / _safe_sqrt(T1T01)

    # Relative Mach at 1 (divide by cos(beta1s))
    AM1R = AM1 / _guard(cb1s)

    # Relative Mach ratio (inlet/discharge)
    AMR = AM1R / _guard(AM2R)

    # Relative speed magnitudes normalized by A01
    W1A01 = _safe_sqrt(WT1A01 ** 2 + C1A01 ** 2)
    # W2/A01 = (W2/a2) * (a2/A01) = AM2R * sqrt(T2/T01) ; here sqrt(T2/T01) = sqrt(TO2TO1 / TO2T2)
    W2A01 = AM2R * _safe_sqrt(TO2TO1 / _guard(TO2T2))

    WR = W1A01 / _guard(W2A01)

    # Density ratio at inlet (static vs stagnation)
    # rho/rho0 = 1 / (1 + (gamma-1)/2 * M^2)^(1/(gamma-1))
    R1RO1 = 1.0 / ((1.0 + 0.5 * (P.G - 1.0) * (AM1 ** 2)) ** (1.0 / (P.G - 1.0)))

    # Head/flow coefficients
    PSI = 2.0 * AL * P.ETS
    PHI = R1RO1 * (P.R1SR2 ** 2) * (1.0 - P.ANU ** 2) * (C1A01 / _guard(U2A01))
    THETA = PHI * U2A01

    # Discharge blade height ratio b2/r2 from continuity: theta = 2*(b2/r2)*(rho2/rho01)*(Cm2/A01)
    B2R2 = THETA / (2.0 * _guard(R2RO1) * _guard(CM2A01))

    # Specific speeds (stagnation and static) – consistent with BASIC structure:
    # Ns,stag ~ sqrt(pi*PHI) / ( (PSI/2)^(3/4) )
    SS  = sqrt(pi * PHI) / ((_guard(PSI) / 2.0) ** 0.75) if isfinite(PHI) and isfinite(PSI) else float('nan')
    # Ns,static uses phi based on inlet static density (omit R1RO1 per BASIC lines 750–760 pattern)
    phi_static_like = (P.R1SR2 ** 2) * (1.0 - P.ANU ** 2) * (C1A01 / _guard(U2A01))
    SSG = sqrt(pi * phi_static_like) / ((_guard(AL * P.ETS)) ** 0.75) if isfinite(phi_static_like) and isfinite(AL) else float('nan')

    # Non-dimensional power coefficient (product form as in BASIC)
    WND = PSI * THETA * (U2A01 ** 2) if isfinite(PSI) and isfinite(THETA) and isfinite(U2A01) else float('nan')

    r2_mm = b2_mm = r1s_mm = r1h_mm = None

    # Optional geometric sizing if inputs present
    if S.mf is not None and S.p01_bar is not None and S.t01 is not None:
        p01_pa = S.p01_bar * 1e5
        rho01 = p01_pa / (P.Rgas * S.t01)
        A01 = sqrt(P.G * P.Rgas * S.t01)   # speed of sound at stagnation T01
        # From BASIC: A2 = mf / (theta * rho01 * A01)
        A2 = S.mf / (_guard(THETA) * rho01 * _guard(A01))
        r2 = _safe_sqrt(A2 / pi)
        b2 = B2R2 * r2
        r1s = P.R1SR2 * r2
        r1h = P.ANU * r1s
        # Convert to mm for printing
        r2_mm  = r2 * 1000.0
        b2_mm  = b2 * 1000.0
        r1s_mm = r1s * 1000.0
        r1h_mm = r1h * 1000.0

    return ResultRow(
        BETAB2=betab2_deg, B2R2=B2R2, U2A01=U2A01, AM2=AM2, AM2R=AM2R,
        AM1=AM1, AM1R=AM1R, AMR=AMR, WR=WR, SS=SS, SSG=SSG, THETA=THETA,
        PHI=PHI, PSI=PSI, WND=WND, r2_mm=r2_mm, b2_mm=b2_mm,
        r1s_mm=r1s_mm, r1h_mm=r1h_mm
    )

def design_impeller(pr: float, beta2_start: float, beta2_step: float, nsteps: int,
                    params: Params | None = None,
                    sizing: SizingInputs | None = None) -> List[ResultRow]:
    P = params or Params()
    S = sizing or SizingInputs()
    out: List[ResultRow] = []
    for i in range(nsteps):
        betab2 = beta2_start - i * beta2_step  # BASIC loop: start, then decrement by step for NSTEP steps
        out.append(design_step(pr, betab2, P, S))
    return out

def _fmt(x: Optional[float], wid=9, prec=5) -> str:
    if x is None: 
        return " " * wid
    if not isfinite(x):
        return " " * (wid - 3) + "nan"
    return f"{x:>{wid}.{prec}f}"

def print_table(rows: List[ResultRow], include_dims: bool = False) -> None:
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
            print(f"    -> r2={r.r2_mm:.1f} mm,  b2={r.b2_mm:.1f} mm,  r1s={r.r1s_mm:.1f} mm,  r1h={r.r1h_mm:.1f} mm")

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Baines BASIC compressor impeller design (Python translation)")
    ap.add_argument("--pr", type=float, required=True, help="Design pressure ratio (stage Po2/Po1)")
    ap.add_argument("--beta2-start", type=float, required=True, help="Starting impeller discharge blade metal angle, deg")
    ap.add_argument("--beta2-step", type=float, required=True, help="Step length for blade angle, deg (positive)")
    ap.add_argument("--nsteps", type=int, required=True, help="Number of steps")
    # Overrides for defaults (optional)
    ap.add_argument("--sf", type=float, default=Params.SF, help="Slip factor")
    ap.add_argument("--eti", type=float, default=Params.ETI, help="Impeller total-to-total efficiency")
    ap.add_argument("--ets", type=float, default=Params.ETS, help="Stage total-to-total efficiency")
    ap.add_argument("--beta1s", type=float, default=Params.BETA1S, help="Inlet shroud relative flow angle (deg)")
    ap.add_argument("--anu", type=float, default=Params.ANU, help="Inducer hub-to-shroud radius ratio r1h/r1s")
    ap.add_argument("--alp2", type=float, default=Params.ALP2, help="Discharge absolute flow angle (deg)")
    ap.add_argument("--r1sr2", type=float, default=Params.R1SR2, help="Inducer shroud to discharge tip radius ratio r1s/r2")
    ap.add_argument("--g", type=float, default=Params.G, help="Ratio of specific heats gamma")
    # Optional physical sizing inputs
    ap.add_argument("--mf", type=float, help="Mass flow rate [kg/s] (optional)")
    ap.add_argument("--p01bar", type=float, help="Inlet stagnation pressure [bar] (optional)")
    ap.add_argument("--t01", type=float, help="Inlet stagnation temperature [K] (optional)")
    args = ap.parse_args()

    P = Params(SF=args.sf, ETI=args.eti, ETS=args.ets, BETA1S=args.beta1s, ANU=args.anu,
               ALP2=args.alp2, R1SR2=args.r1sr2, G=args.g)
    S = SizingInputs(mf=args.mf, p01_bar=args.p01bar, t01=args.t01)

    rows = design_impeller(args.pr, args.beta2_start, args.beta2_step, args.nsteps, P, S)
    include_dims = (S.mf is not None and S.p01_bar is not None and S.t01 is not None)
    print("\nDESIGN PRESSURE RATIO:", args.pr)
    print("Parameters:",
          f"SF={P.SF}, ETI={P.ETI}, ETS={P.ETS}, BETA1S={P.BETA1S}°, ANU={P.ANU}, ALP2={P.ALP2}°, R1SR2={P.R1SR2}, G={P.G}")
    if include_dims:
        print(f"Sizing inputs: mf={S.mf} kg/s, p01={S.p01_bar} bar, t01={S.t01} K")
    print()
    print_table(rows, include_dims=include_dims)

if __name__ == "__main__":
    main()
