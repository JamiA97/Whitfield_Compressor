"""
Matplotlib plot helpers embedded in Qt widgets.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget


PLOT_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("b2/r2 vs beta2", "beta_b2", "b2/r2"),
    ("M2_rel vs beta2", "beta_b2", "M2_rel"),
    ("Ns(stag) vs beta2", "beta_b2", "Ns(stag)"),
    ("THETA vs beta2", "beta_b2", "THETA"),
)


class PlotTabWidget(QTabWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._axes: list = []
        self._canvases: list[FigureCanvasQTAgg] = []
        for title, _, _ in PLOT_SPECS:
            widget = QWidget(self)
            layout = QVBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            fig = Figure(figsize=(4, 3), tight_layout=True)
            canvas = FigureCanvasQTAgg(fig)
            toolbar = NavigationToolbar2QT(canvas, widget)
            layout.addWidget(toolbar)
            layout.addWidget(canvas)
            ax = fig.add_subplot(111)
            ax.grid(True, linestyle=":", linewidth=0.6)
            self.addTab(widget, title)
            self._axes.append(ax)
            self._canvases.append(canvas)

    def update_plots(self, df: pd.DataFrame) -> None:
        for (title, x_col, y_col), ax, canvas in zip(PLOT_SPECS, self._axes, self._canvases):
            ax.clear()
            ax.grid(True, linestyle=":", linewidth=0.6)
            ax.set_title(title)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            if not df.empty and x_col in df.columns and y_col in df.columns:
                ax.plot(df[x_col], df[y_col], marker="o")
            canvas.draw_idle()
