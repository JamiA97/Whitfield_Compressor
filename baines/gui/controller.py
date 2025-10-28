"""
Application controller wiring UI events to the computational core.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from PySide6.QtCore import QObject
from PySide6.QtWidgets import (
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QStatusBar,
    QWidget,
)

from baines.core import Params, ResultRow, SizingInputs, design_impeller

from .plots import PlotTabWidget
from .widgets import FieldSpec, ParameterPanel, ResultsTableView


LOGGER = logging.getLogger("baines.gui")


def _config_root() -> Path:
    home = Path.home()
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = home / "Library" / "Application Support"
    else:
        base = home / ".config"
    return base / "baines_gui"


def ensure_logger() -> None:
    if LOGGER.handlers:
        return
    log_dir = _config_root()
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / "baines_gui.log", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def _read_simple_yaml(path: Path) -> Dict[str, float | str]:
    data: Dict[str, float | str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" not in stripped:
                continue
            key, _, value = stripped.partition(":")
            key = key.strip()
            value_str = value.strip()
            # remove inline comments
            if " #" in value_str:
                value_str = value_str.split(" #", 1)[0].strip()
            if value_str.startswith("'") and value_str.endswith("'"):
                data[key] = value_str[1:-1]
                continue
            try:
                if "." in value_str or "e" in value_str.lower():
                    data[key] = float(value_str)
                else:
                    data[key] = float(int(value_str))
            except ValueError:
                LOGGER.warning("Skipping unparsable preset value '%s' for key '%s'", value_str, key)
    return data


def _write_simple_yaml(path: Path, data: Dict[str, float | str]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for key, value in sorted(data.items()):
            if isinstance(value, str):
                fh.write(f"{key}: '{value}'\n")
            else:
                fh.write(f"{key}: {value}\n")


def compute_warnings(df: pd.DataFrame) -> pd.Series:
    msgs: List[str] = []
    for _, row in df.iterrows():
        warnings: List[str] = []
        if row.get("M2_rel", 0.0) > 1.2:
            warnings.append("M2_rel>1.2")
        b2r2 = row.get("b2/r2")
        if pd.notna(b2r2) and not (0.03 <= b2r2 <= 0.12):
            warnings.append("b2/r2 out of band")
        if any(pd.isna(row.get(col)) for col in ("U2/A0", "PHI", "PSI")):
            warnings.append("NaN fields")
        msgs.append("; ".join(warnings))
    return pd.Series(msgs, index=df.index, name="Warnings")


def build_dataframe(rows: List[ResultRow]) -> Tuple[pd.DataFrame, Dict[int, str]]:
    records: List[Dict[str, float | str]] = []
    tooltips: Dict[int, str] = {}
    for idx, row in enumerate(rows):
        rec = {
            "beta_b2": row.BETAB2,
            "b2/r2": row.B2R2,
            "U2/A0": row.U2A01,
            "M2": row.AM2,
            "M2_rel": row.AM2R,
            "M1": row.AM1,
            "M1_rel": row.AM1R,
            "M1r/M2r": row.AMR,
            "W1/W2": row.WR,
            "Ns(stag)": row.SS,
            "Ns(static)": row.SSG,
            "THETA": row.THETA,
            "PHI": row.PHI,
            "PSI": row.PSI,
            "WND": row.WND,
            "r2 [mm]": row.r2_mm,
            "b2 [mm]": row.b2_mm,
            "r1s [mm]": row.r1s_mm,
            "r1h [mm]": row.r1h_mm,
        }
        records.append(rec)
        if row.r2_mm is not None:
            tooltips[idx] = (
                f"r2={row.r2_mm:.1f} mm, b2={row.b2_mm:.1f} mm, "
                f"r1s={row.r1s_mm:.1f} mm, r1h={row.r1h_mm:.1f} mm"
            )
    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df.insert(len(df.columns), "Warnings", compute_warnings(df))
    else:
        df["Warnings"] = pd.Series(dtype=str)
    return df, tooltips


DESIGN_SPECS = [
    FieldSpec("pr", "PR", 2.5, minimum=1.05, maximum=6.0, tooltip="Stage total pressure ratio"),
    FieldSpec(
        "beta2_start",
        "beta2_start [deg]",
        0.0,
        minimum=-80.0,
        maximum=80.0,
        tooltip="Starting discharge blade metal angle",
    ),
    FieldSpec(
        "beta2_step",
        "beta2_step [deg]",
        10.0,
        minimum=0.1,
        maximum=30.0,
        tooltip="Positive decrement per step",
    ),
    FieldSpec(
        "nsteps",
        "nsteps",
        5,
        minimum=1,
        maximum=200,
        tooltip="Number of beta2 sweep steps",
        kind="int",
    ),
]

PARAM_SPECS = [
    FieldSpec("SF", "SF", Params.SF, minimum=0.5, maximum=1.0, tooltip="Slip factor"),
    FieldSpec("ETI", "ETI", Params.ETI, minimum=0.5, maximum=0.95, tooltip="Impeller efficiency"),
    FieldSpec("ETS", "ETS", Params.ETS, minimum=0.5, maximum=0.95, tooltip="Stage efficiency"),
    FieldSpec(
        "BETA1S",
        "BETA1S [deg]",
        Params.BETA1S,
        minimum=-85.0,
        maximum=0.0,
        tooltip="Inlet shroud relative flow angle",
    ),
    FieldSpec(
        "ANU",
        "ANU",
        Params.ANU,
        minimum=0.2,
        maximum=0.8,
        tooltip="Inducer hub/shroud radius ratio",
    ),
    FieldSpec(
        "ALP2",
        "ALP2 [deg]",
        Params.ALP2,
        minimum=20.0,
        maximum=80.0,
        tooltip="Discharge absolute flow angle",
    ),
    FieldSpec(
        "R1SR2",
        "R1SR2",
        Params.R1SR2,
        minimum=0.4,
        maximum=0.9,
        tooltip="Inducer shroud to discharge tip radius ratio",
    ),
    FieldSpec("G", "gamma", Params.G, minimum=1.2, maximum=1.7, tooltip="Specific heat ratio"),
]

SIZING_SPECS = [
    FieldSpec("mf", "mf [kg/s]", None, minimum=0.01, maximum=100.0, tooltip="Mass flow rate", required=False),
    FieldSpec(
        "p01_bar",
        "p01 [bar]",
        None,
        minimum=0.2,
        maximum=5.0,
        tooltip="Inlet stagnation pressure",
        required=False,
    ),
    FieldSpec(
        "t01",
        "t01 [K]",
        None,
        minimum=150.0,
        maximum=800.0,
        tooltip="Inlet stagnation temperature",
        required=False,
    ),
]


class CompressorController(QObject):
    def __init__(
        self,
        panel: ParameterPanel,
        table: ResultsTableView,
        plots: PlotTabWidget,
        status_bar: QStatusBar,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        ensure_logger()
        self.panel = panel
        self.table = table
        self.plots = plots
        self.status_bar = status_bar
        self._builtin_presets = self._discover_builtin_presets()
        self.panel.set_presets(self._collect_preset_names())
        self._last_df = pd.DataFrame()
        self._last_tooltips: Dict[int, str] = {}
        self._last_rows: List[ResultRow] = []
        self._last_inputs: Dict[str, Dict[str, float | None]] = {}

        self.panel.runRequested.connect(self.on_run_clicked)
        self.panel.resetRequested.connect(self.on_reset_clicked)
        self.panel.savePresetRequested.connect(self.on_save_preset)
        self.panel.loadPresetRequested.connect(self.on_load_preset)
        self.panel.presetSelected.connect(self.on_preset_selected)
        self.table.exportRequested.connect(self.on_export_selected_csv)

    # region Presets
    def _builtin_preset_dir(self) -> Path:
        return Path(__file__).resolve().parent / "presets"

    def _discover_builtin_presets(self) -> Dict[str, Dict[str, float]]:
        presets: Dict[str, Dict[str, float]] = {}
        preset_dir = self._builtin_preset_dir()
        if not preset_dir.exists():
            return presets
        for path in preset_dir.glob("*.yml"):
            presets[path.stem] = _read_simple_yaml(path)
        return presets

    def builtin_preset_names(self) -> List[str]:
        return sorted(self._builtin_presets.keys())

    def _user_preset_dir(self) -> Path:
        path = _config_root() / "presets"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _collect_preset_names(self) -> List[str]:
        names = set(self._builtin_presets.keys())
        for path in self._user_preset_dir().glob("*.yml"):
            names.add(path.stem)
        return sorted(names)

    def on_preset_selected(self, name: str) -> None:
        if name == "Custom":
            return
        self.panel.preset_combo.blockSignals(True)
        idx = self.panel.preset_combo.findText(name)
        if idx >= 0:
            self.panel.preset_combo.setCurrentIndex(idx)
        else:
            self.panel.preset_combo.setCurrentIndex(0)
        self.panel.preset_combo.blockSignals(False)
        data = self._builtin_presets.get(name)
        if data is None:
            user_path = self._user_preset_dir() / f"{name}.yml"
            if user_path.exists():
                data = _read_simple_yaml(user_path)
            else:
                QMessageBox.warning(None, "Preset not found", f"No preset named '{name}'")
                return
        self._apply_preset(data)

    def _apply_preset(self, data: Dict[str, float]) -> None:
        self.panel.apply_values(data)

    def on_save_preset(self) -> None:
        name, ok = QInputDialog.getText(None, "Save Preset", "Preset name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        values = {k: v for k, v in self.panel.values().items() if v is not None}
        path = self._user_preset_dir() / f"{name}.yml"
        _write_simple_yaml(path, values)  # type: ignore[arg-type]
        QMessageBox.information(None, "Preset saved", f"Saved preset '{name}' to {path}")
        self.panel.set_presets(self._collect_preset_names())

    def on_load_preset(self) -> None:
        directory = str(self._user_preset_dir())
        path, _ = QFileDialog.getOpenFileName(
            None,
            "Load Preset",
            directory,
            "Preset files (*.yml);;All files (*)",
        )
        if not path:
            return
        data = _read_simple_yaml(Path(path))
        self._apply_preset(data)

    # endregion

    def show_user_preset_location(self) -> None:
        path = self._user_preset_dir()
        QMessageBox.information(
            None,
            "Preset location",
            f"User presets are saved under:\n{path}",
        )

    # region Run workflow
    def on_reset_clicked(self) -> None:
        self.panel.reset_defaults()
        self.status_bar.showMessage("Inputs reset to defaults", 3000)

    def validate_inputs(self, values: Dict[str, float | None]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float | None], List[str]]:
        errors: List[str] = []
        design = {
            "pr": values.get("pr"),
            "beta2_start": values.get("beta2_start"),
            "beta2_step": values.get("beta2_step"),
            "nsteps": values.get("nsteps"),
        }
        params = {
            "SF": values.get("SF"),
            "ETI": values.get("ETI"),
            "ETS": values.get("ETS"),
            "BETA1S": values.get("BETA1S"),
            "ANU": values.get("ANU"),
            "ALP2": values.get("ALP2"),
            "R1SR2": values.get("R1SR2"),
            "G": values.get("G"),
        }
        sizing = {
            "mf": values.get("mf"),
            "p01_bar": values.get("p01_bar"),
            "t01": values.get("t01"),
        }

        for key, val in design.items():
            if val is None:
                errors.append(key)

        beta2_step = design["beta2_step"]
        if beta2_step is not None and beta2_step <= 0:
            errors.append("beta2_step")
        if design["pr"] is not None and design["pr"] <= 1.0:
            errors.append("pr")
        if design["nsteps"] is not None:
            design["nsteps"] = float(int(design["nsteps"]))
            if design["nsteps"] < 1:
                errors.append("nsteps")
        alp2 = params["ALP2"]
        if alp2 is not None and abs(alp2 - 90.0) < 1e-3:
            errors.append("ALP2")

        if any(value is not None for value in sizing.values()):
            if sizing["mf"] is None:
                errors.append("mf")
            if sizing["p01_bar"] is None:
                errors.append("p01_bar")
            if sizing["t01"] is None:
                errors.append("t01")

        design_clean = {k: float(v) for k, v in design.items() if v is not None}
        params_clean = {k: float(v) for k, v in params.items() if v is not None}

        return design_clean, params_clean, sizing, errors

    def on_run_clicked(self) -> None:
        try:
            self.panel.mark_errors([])
            values = self.panel.values()
            design, params, sizing, errors = self.validate_inputs(values)
            if errors:
                self.panel.mark_errors(errors)
                QMessageBox.warning(None, "Invalid inputs", "Please correct highlighted fields.")
                return
            start = time.perf_counter()
            rows = design_impeller(
                design["pr"],
                design["beta2_start"],
                design["beta2_step"],
                int(design["nsteps"]),
                Params(**params),
                SizingInputs(**{k: (float(v) if v is not None else None) for k, v in sizing.items()}),
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            df, tooltips = build_dataframe(rows)
            self.table.set_dataframe(df, tooltips)
            self.plots.update_plots(df)
            warn_count = int((df["Warnings"] != "").sum()) if "Warnings" in df.columns else 0
            self.status_bar.showMessage(
                f"Run complete | Rows: {len(df)} | Warnings: {warn_count} | Runtime: {elapsed_ms:.1f} ms"
            )
            self._last_df = df
            self._last_tooltips = tooltips
            self._last_rows = rows
            self._last_inputs = {
                "design": design,
                "params": params,
                "sizing": sizing,
            }
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Run failed")
            QMessageBox.critical(None, "Run failed", str(exc))

    # endregion

    # region Export/session
    def on_export_selected_csv(self, rows: List[int]) -> None:
        self._export_csv(rows)

    def export_full_csv(self) -> None:
        self._export_csv(None)

    def _export_csv(self, rows: List[int] | None) -> None:
        if self._last_df.empty:
            QMessageBox.information(None, "No data", "Run the solver before exporting.")
            return
        df = self._last_df
        if rows is not None:
            df = df.iloc[rows]
        path, _ = QFileDialog.getSaveFileName(
            None,
            "Export CSV",
            str(Path.home() / "baines_results.csv"),
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        df.to_csv(path, index=False)
        self.status_bar.showMessage(f"Exported CSV to {path}", 5000)

    def export_json(self) -> None:
        if self._last_df.empty:
            QMessageBox.information(None, "No data", "Run the solver before exporting.")
            return
        path, _ = QFileDialog.getSaveFileName(
            None,
            "Export JSON",
            str(Path.home() / "baines_results.json"),
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        header = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "inputs": self._last_inputs,
        }
        payload = {
            "header": header,
            "rows": [asdict(row) for row in self._last_rows],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        self.status_bar.showMessage(f"Exported JSON to {path}", 5000)

    def save_session(self) -> None:
        if not self._last_inputs:
            QMessageBox.information(None, "No session", "Run the solver before saving a session.")
            return
        path, _ = QFileDialog.getSaveFileName(
            None,
            "Save Session",
            str(Path.home() / "session.baines.yml"),
            "Baines session (*.baines.yml);;All files (*)",
        )
        if not path:
            return
        data = {
            "inputs": self._last_inputs,
            "results": [asdict(row) for row in self._last_rows],
            "tooltips": self._last_tooltips,
        }
        _write_simple_yaml(Path(path), {"_json": json.dumps(data)})
        self.status_bar.showMessage(f"Session saved to {path}", 5000)

    def load_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            None,
            "Load Session",
            str(Path.home()),
            "Baines session (*.baines.yml);;All files (*)",
        )
        if not path:
            return
        data = _read_simple_yaml(Path(path))
        json_blob = data.get("_json")
        if json_blob is None:
            QMessageBox.warning(None, "Invalid session", "Session file missing payload.")
            return
        session = json.loads(json_blob)
        inputs = session.get("inputs", {})
        design = inputs.get("design", {})
        params = inputs.get("params", {})
        sizing = inputs.get("sizing", {})
        merged = {**design, **params, **sizing}
        self.panel.apply_values(merged)
        rows_json = session.get("results", [])
        rows: List[ResultRow] = []
        for item in rows_json:
            try:
                rows.append(ResultRow(**item))
            except TypeError:
                LOGGER.warning("Skipping malformed result row in session")
        df, tooltips = build_dataframe(rows)
        tooltip_payload = session.get("tooltips", {})
        if tooltip_payload:
            tooltips = {int(k): v for k, v in tooltip_payload.items()}
        self.table.set_dataframe(df, tooltips)
        self.plots.update_plots(df)
        self._last_df = df
        self._last_tooltips = tooltips
        self._last_rows = rows
        self._last_inputs = inputs
        self.status_bar.showMessage(f"Loaded session from {path}", 5000)

    # endregion
