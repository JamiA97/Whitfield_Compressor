"""
Qt widgets for the Baines GUI application.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QSortFilterProxyModel, Qt, Signal
from PySide6.QtGui import QDoubleValidator, QGuiApplication, QIntValidator, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableView,
    QVBoxLayout,
    QWidget,
)


def _format_float(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.4g}"


@dataclass
class FieldSpec:
    key: str
    label: str
    default: float | int | None
    minimum: float | None = None
    maximum: float | None = None
    tooltip: str | None = None
    kind: str = "float"  # float or int
    required: bool = True


class FloatLineEdit(QLineEdit):
    def __init__(self, spec: FieldSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self.setAlignment(Qt.AlignmentFlag.AlignRight)
        if spec.kind == "float":
            validator = QDoubleValidator(parent=self)
            if spec.minimum is not None:
                validator.setBottom(spec.minimum)
            if spec.maximum is not None:
                validator.setTop(spec.maximum)
            validator.setNotation(QDoubleValidator.Notation.StandardNotation)
            self.setValidator(validator)
        elif spec.kind == "int":
            validator = QIntValidator(parent=self)
            if spec.minimum is not None:
                validator.setBottom(int(spec.minimum))
            if spec.maximum is not None:
                validator.setTop(int(spec.maximum))
            self.setValidator(validator)
        if spec.default is not None:
            self.setText(str(spec.default))
        if spec.tooltip:
            self.setToolTip(spec.tooltip)

    def value(self) -> Optional[float]:
        text = self.text().strip()
        if not text:
            return None
        if self.spec.kind == "int":
            return float(int(text))
        return float(text)

    def set_value(self, value: Optional[float]) -> None:
        if value is None:
            self.clear()
        else:
            if self.spec.kind == "int":
                self.setText(str(int(value)))
            else:
                self.setText(str(value))


class ParameterPanel(QWidget):
    runRequested = Signal()
    resetRequested = Signal()
    savePresetRequested = Signal()
    loadPresetRequested = Signal()
    presetSelected = Signal(str)

    def __init__(
        self,
        design_specs: Iterable[FieldSpec],
        param_specs: Iterable[FieldSpec],
        sizing_specs: Iterable[FieldSpec],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._design_specs = list(design_specs)
        self._param_specs = list(param_specs)
        self._sizing_specs = list(sizing_specs)
        self._fields: Dict[str, FloatLineEdit | QSpinBox] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        layout.addWidget(self._build_group("Design Inputs", self._design_specs))
        layout.addWidget(self._build_group("Model Parameters", self._param_specs))
        layout.addWidget(self._build_group("Optional Sizing", self._sizing_specs))

        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox(self)
        preset_layout.addWidget(self.preset_combo, 1)
        layout.addLayout(preset_layout)
        self.preset_combo.currentIndexChanged.connect(self._emit_preset)

        button_row = QHBoxLayout()
        self.run_btn = QPushButton("Run", self)
        self.reset_btn = QPushButton("Reset", self)
        self.save_preset_btn = QPushButton("Save Preset", self)
        self.load_preset_btn = QPushButton("Load Preset", self)
        for btn in (self.run_btn, self.reset_btn, self.save_preset_btn, self.load_preset_btn):
            button_row.addWidget(btn)
        layout.addLayout(button_row)
        layout.addStretch()

        self.run_btn.clicked.connect(self.runRequested)
        self.reset_btn.clicked.connect(self.resetRequested)
        self.save_preset_btn.clicked.connect(self.savePresetRequested)
        self.load_preset_btn.clicked.connect(self.loadPresetRequested)

    def _build_group(self, title: str, specs: Iterable[FieldSpec]) -> QGroupBox:
        box = QGroupBox(title, self)
        form = QFormLayout(box)
        for spec in specs:
            if spec.kind == "int":
                spin = QSpinBox(box)
                if spec.minimum is not None:
                    spin.setMinimum(int(spec.minimum))
                if spec.maximum is not None:
                    spin.setMaximum(int(spec.maximum))
                else:
                    spin.setMaximum(10_000)
                spin.setAlignment(Qt.AlignmentFlag.AlignRight)
                if spec.default is not None:
                    spin.setValue(int(spec.default))
                if spec.tooltip:
                    spin.setToolTip(spec.tooltip)
                form.addRow(spec.label + ":", spin)
                self._fields[spec.key] = spin
            else:
                fld = FloatLineEdit(spec, box)
                form.addRow(spec.label + ":", fld)
                self._fields[spec.key] = fld
        return box

    def set_presets(self, entries: List[str]) -> None:
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("Custom", userData=None)
        for entry in entries:
            self.preset_combo.addItem(entry)
        self.preset_combo.blockSignals(False)

    def _emit_preset(self, index: int) -> None:
        if index <= 0:
            return
        data = self.preset_combo.itemData(index)
        if data is None:
            text = self.preset_combo.itemText(index)
        else:
            text = data
        if text:
            self.presetSelected.emit(text)

    def values(self) -> Dict[str, Optional[float]]:
        result: Dict[str, Optional[float]] = {}
        for key, widget in self._fields.items():
            if isinstance(widget, QSpinBox):
                result[key] = float(widget.value())
            else:
                result[key] = widget.value()
        return result

    def apply_values(self, values: Dict[str, float | None]) -> None:
        for key, value in values.items():
            widget = self._fields.get(key)
            if widget is None:
                continue
            if isinstance(widget, QSpinBox):
                if value is not None:
                    widget.setValue(int(value))
            else:
                widget.set_value(value)

    def reset_defaults(self) -> None:
        for spec in (*self._design_specs, *self._param_specs, *self._sizing_specs):
            widget = self._fields.get(spec.key)
            if widget is None:
                continue
            if isinstance(widget, QSpinBox):
                if spec.default is not None:
                    widget.setValue(int(spec.default))
                else:
                    widget.setValue(widget.minimum())
            else:
                widget.set_value(spec.default if spec.required else None)

    def mark_errors(self, error_keys: Iterable[str]) -> None:
        err_set = set(error_keys)
        for key, widget in self._fields.items():
            if isinstance(widget, QSpinBox):
                widget.setStyleSheet("border: 1px solid #d9534f;" if key in err_set else "")
            else:
                widget.setStyleSheet("border: 1px solid #d9534f;" if key in err_set else "")


class PandasTableModel(QAbstractTableModel):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._df = pd.DataFrame()
        self._tooltips: Dict[int, str] = {}

    def set_dataframe(self, df: pd.DataFrame, tooltips: Dict[int, str] | None = None) -> None:
        self.beginResetModel()
        self._df = df
        self._tooltips = tooltips or {}
        self.endResetModel()

    def rowCount(self, parent=None) -> int:  # type: ignore[override]
        return 0 if parent and parent.isValid() else len(self._df.index)

    def columnCount(self, parent=None) -> int:  # type: ignore[override]
        return 0 if parent and parent.isValid() else len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):  # type: ignore[override]
        if not index.isValid():
            return None
        value = self._df.iat[index.row(), index.column()]
        if role == Qt.DisplayRole:
            if isinstance(value, float):
                return _format_float(value)
            return str(value)
        if role == Qt.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        if role == Qt.ToolTipRole:
            return self._tooltips.get(index.row())
        if role == Qt.UserRole:
            return value
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.DisplayRole):  # type: ignore[override]
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self._df.columns[section]
        return str(self._df.index[section])


class ResultsTableView(QTableView):
    exportRequested = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = PandasTableModel(self)
        self._proxy = QSortFilterProxyModel(self)
        self._proxy.setSourceModel(self._model)
        self._proxy.setSortRole(Qt.UserRole)
        self.setModel(self._proxy)
        self.setSortingEnabled(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setStretchLastSection(False)
        self.verticalHeader().setVisible(False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_dataframe(self, df: pd.DataFrame, tooltips: Dict[int, str] | None = None) -> None:
        self._model.set_dataframe(df, tooltips)
        self.resizeColumnsToContents()

    def current_dataframe(self) -> pd.DataFrame:
        return self._model._df.copy()

    def selected_source_rows(self) -> List[int]:
        rows: List[int] = []
        for index in self.selectionModel().selectedRows():
            source = self._proxy.mapToSource(index)
            rows.append(source.row())
        return rows

    def copy_selection_to_clipboard(self) -> None:
        rows = sorted(set(self.selected_source_rows()))
        if not rows:
            return
        df = self._model._df.iloc[rows]
        text = df.to_csv(index=False, sep="\t")
        QGuiApplication.clipboard().setText(text)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.matches(QKeySequence.StandardKey.Copy):
            self.copy_selection_to_clipboard()
            event.accept()
            return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event) -> None:  # type: ignore[override]
        menu = QMenu(self)
        export_action = menu.addAction("Export selected rows to CSV…")
        action = menu.exec(event.globalPos())
        if action == export_action:
            rows = self.selected_source_rows()
            if rows:
                self.exportRequested.emit(rows)
