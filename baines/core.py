"""
Core computational routines for the Baines centrifugal compressor sizing tool.

This module contains the pure calculation layer migrated from the original
`baines_compressor.py` script so it can be reused by both CLI and GUI front ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan, cos, degrees, isfinite, pi, radians, sin, sqrt, tan
from typing import List, Optional


@dataclass
class Params:
    SF: float = 0.85  # slip factor
    ETI: float = 0.90  # impeller total-to-total efficiency
    ETS: float = 0.80  # stage total-to-total efficiency
    BETA1S: float = -60.0  # inlet shroud relative flow angle [deg]
    ANU: float = 0.40  # r1h / r1s (hub/shroud radius ratio at inducer)
    ALP2: float = 65.0  # discharge absolute flow angle [deg]
    R1SR2: float = 0.70  # r1s / r2 (inducer shroud to discharge tip radius ratio)
    G: float = 1.40  # ratio of specific heats (gamma)
    Rgas: float = 287.0  # specific gas constant for air [J/(kg*K)]


@dataclass
class SizingInputs:
    mf: Optional[float] = None  # mass flow rate [kg/s]
    p01_bar: Optional[float] = None  # inlet stagnation pressure [bar]
    t01: Optional[float] = None  # inlet stagnation temperature [K]


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
    r2_mm: Optional[float] = None
    b2_mm: Optional[float] = None
    r1s_mm: Optional[float] = None
    r1h_mm: Optional[float] = None


def _safe_sqrt(x: float) -> float:
    return sqrt(x) if x > 0.0 else float("nan")


def _guard(x: float, eps: float = 1e-12) -> float:
    if abs(x) < eps:
        return float("nan")
    return x


def design_step(pr: float, betab2_deg: float, P: Params, S: SizingInputs) -> ResultRow:
    beta1 = radians(P.BETA1S)
    tb1s = tan(beta1)
    cb1s = cos(beta1)
    alp2r = radians(P.ALP2)
    sa2 = sin(alp2r)
    ca2 = cos(alp2r)
    ta2 = tan(alp2r)

    betb2 = radians(betab2_deg)
    tbb2 = tan(betb2)

    denom = 1.0 - (tbb2 / _guard(ta2))
    AL = P.SF / denom if isfinite(denom) else float("nan")

    pr_term = pr ** ((P.G - 1.0) / P.G) - 1.0
    denom_u = (P.G - 1.0) * P.ETS * AL
    U2A01 = _safe_sqrt(pr_term / denom_u) if isfinite(denom_u) else float("nan")

    CT2A01 = U2A01 * AL

    TO2TO1 = 1.0 + (P.G - 1.0) * U2A01 * CT2A01

    PO2PO1 = (1.0 + P.ETI * (TO2TO1 - 1.0)) ** (P.G / (P.G - 1.0))

    C2A01 = CT2A01 / _guard(sa2)
    C2A02 = C2A01 / _safe_sqrt(TO2TO1)
    T2TO2 = 1.0 - 0.5 * (P.G - 1.0) * (C2A02 ** 2)
    TO2T2 = 1.0 / T2TO2 if isfinite(T2TO2) else float("nan")

    AM2 = C2A02 * _safe_sqrt(TO2T2)

    U2A02 = U2A01 / _safe_sqrt(TO2TO1)
    U2A2 = U2A02 * _safe_sqrt(T2TO2)
    CT2A2 = CT2A01 * _safe_sqrt(T2TO2 / TO2TO1) if isfinite(TO2TO1) else float("nan")

    WT2A2 = CT2A2 - U2A2
    CM2A2 = AM2 * ca2

    AM2R = _safe_sqrt(WT2A2 ** 2 + CM2A2 ** 2)

    B2 = degrees(atan(WT2A2 / _guard(CM2A2)))

    PO2P2 = (TO2T2) ** (P.G / (P.G - 1.0)) if isfinite(TO2T2) else float("nan")
    P2PO1 = PO2PO1 / _guard(PO2P2)

    R2RO1 = P2PO1 * (T2TO2 / _guard(TO2TO1))

    CM2A01 = CT2A01 / _guard(ta2)

    U1A01 = U2A01 * P.R1SR2

    WT1A01 = -U1A01
    C1A01 = WT1A01 / _guard(tb1s)

    T1T01 = 1.0 - 0.5 * (P.G - 1.0) * (C1A01 ** 2)
    AM1 = C1A01 / _safe_sqrt(T1T01)

    AM1R = AM1 / _guard(cb1s)

    AMR = AM1R / _guard(AM2R)

    W1A01 = _safe_sqrt(WT1A01 ** 2 + C1A01 ** 2)
    W2A01 = AM2R * _safe_sqrt(TO2TO1 / _guard(TO2T2))

    WR = W1A01 / _guard(W2A01)

    R1RO1 = 1.0 / ((1.0 + 0.5 * (P.G - 1.0) * (AM1 ** 2)) ** (1.0 / (P.G - 1.0)))

    PSI = 2.0 * AL * P.ETS
    PHI = R1RO1 * (P.R1SR2 ** 2) * (1.0 - P.ANU ** 2) * (C1A01 / _guard(U2A01))
    THETA = PHI * U2A01

    B2R2 = THETA / (2.0 * _guard(R2RO1) * _guard(CM2A01))

    SS = (
        sqrt(pi * PHI) / ((_guard(PSI) / 2.0) ** 0.75)
        if isfinite(PHI) and isfinite(PSI)
        else float("nan")
    )
    phi_static_like = (P.R1SR2 ** 2) * (1.0 - P.ANU ** 2) * (C1A01 / _guard(U2A01))
    SSG = (
        sqrt(pi * phi_static_like) / ((_guard(AL * P.ETS)) ** 0.75)
        if isfinite(phi_static_like) and isfinite(AL)
        else float("nan")
    )

    WND = (
        PSI * THETA * (U2A01 ** 2)
        if isfinite(PSI) and isfinite(THETA) and isfinite(U2A01)
        else float("nan")
    )

    r2_mm = b2_mm = r1s_mm = r1h_mm = None
    if S.mf is not None and S.p01_bar is not None and S.t01 is not None:
        p01_pa = S.p01_bar * 1e5
        rho01 = p01_pa / (P.Rgas * S.t01)
        A01 = sqrt(P.G * P.Rgas * S.t01)
        A2 = S.mf / (_guard(THETA) * rho01 * _guard(A01))
        r2 = _safe_sqrt(A2 / pi)
        b2 = B2R2 * r2
        r1s = P.R1SR2 * r2
        r1h = P.ANU * r1s
        r2_mm = r2 * 1000.0
        b2_mm = b2 * 1000.0
        r1s_mm = r1s * 1000.0
        r1h_mm = r1h * 1000.0

    return ResultRow(
        BETAB2=betab2_deg,
        B2R2=B2R2,
        U2A01=U2A01,
        AM2=AM2,
        AM2R=AM2R,
        AM1=AM1,
        AM1R=AM1R,
        AMR=AMR,
        WR=WR,
        SS=SS,
        SSG=SSG,
        THETA=THETA,
        PHI=PHI,
        PSI=PSI,
        WND=WND,
        r2_mm=r2_mm,
        b2_mm=b2_mm,
        r1s_mm=r1s_mm,
        r1h_mm=r1h_mm,
    )


def design_impeller(
    pr: float,
    beta2_start: float,
    beta2_step: float,
    nsteps: int,
    params: Params | None = None,
    sizing: SizingInputs | None = None,
) -> List[ResultRow]:
    P = params or Params()
    S = sizing or SizingInputs()
    out: List[ResultRow] = []
    for i in range(nsteps):
        betab2 = beta2_start - i * beta2_step
        out.append(design_step(pr, betab2, P, S))
    return out


__all__ = [
    "Params",
    "SizingInputs",
    "ResultRow",
    "design_step",
    "design_impeller",
]
