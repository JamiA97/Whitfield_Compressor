"""
Qt application entry point for the Baines compressor GUI.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSplitter,
)

from .controller import (
    CompressorController,
    DESIGN_SPECS,
    PARAM_SPECS,
    SIZING_SPECS,
)
from .plots import PlotTabWidget
from .widgets import ParameterPanel, ResultsTableView


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Baines Compressor Sizer")
        self.resize(1280, 800)

        self.parameter_panel = ParameterPanel(DESIGN_SPECS, PARAM_SPECS, SIZING_SPECS, self)
        self.results_table = ResultsTableView(self)
        self.plot_tabs = PlotTabWidget(self)

        self._build_layout()
        self.controller = CompressorController(
            self.parameter_panel,
            self.results_table,
            self.plot_tabs,
            self.statusBar(),
            self,
        )
        self._build_menus()
        self._apply_stylesheet()

    def _build_layout(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self.results_table)
        splitter.addWidget(self.plot_tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.setCentralWidget(splitter)

        dock = QDockWidget("Parameters", self)
        dock.setWidget(self.parameter_panel)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        run_action = QAction("Run", self)
        run_action.setShortcut("Ctrl+R")
        run_action.triggered.connect(self.controller.on_run_clicked)
        file_menu.addAction(run_action)

        export_csv_action = QAction("Export CSV…", self)
        export_csv_action.triggered.connect(self.controller.export_full_csv)
        file_menu.addAction(export_csv_action)

        export_json_action = QAction("Export JSON…", self)
        export_json_action.triggered.connect(self.controller.export_json)
        file_menu.addAction(export_json_action)

        file_menu.addSeparator()

        save_session_action = QAction("Save Session…", self)
        save_session_action.triggered.connect(self.controller.save_session)
        file_menu.addAction(save_session_action)

        load_session_action = QAction("Load Session…", self)
        load_session_action.triggered.connect(self.controller.load_session)
        file_menu.addAction(load_session_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        edit_menu = self.menuBar().addMenu("&Edit")
        copy_action = QAction("Copy", self)
        copy_action.setShortcut("Ctrl+C")
        copy_action.triggered.connect(self.results_table.copy_selection_to_clipboard)
        edit_menu.addAction(copy_action)

        presets_menu = self.menuBar().addMenu("&Presets")
        load_builtin_menu = QMenu("Load Built-in", self)
        for name in self.controller.builtin_preset_names():
            action = load_builtin_menu.addAction(name)
            action.triggered.connect(lambda checked=False, n=name: self.controller.on_preset_selected(n))
        presets_menu.addMenu(load_builtin_menu)

        manage_presets_action = QAction("Manage…", self)
        manage_presets_action.triggered.connect(self.controller.show_user_preset_location)
        presets_menu.addAction(manage_presets_action)

        view_menu = self.menuBar().addMenu("&View")
        toggle_params_action = QAction("Toggle Parameters Dock", self)
        toggle_params_action.triggered.connect(self._toggle_parameter_dock)
        view_menu.addAction(toggle_params_action)

        help_menu = self.menuBar().addMenu("&Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _toggle_parameter_dock(self) -> None:
        for dock in self.findChildren(QDockWidget):
            if dock.widget() is self.parameter_panel:
                dock.setVisible(not dock.isVisible())
                break

    def _show_about(self) -> None:
        QMessageBox.information(
            self,
            "About Baines Compressor Sizer",
            "Desktop GUI for the Baines centrifugal compressor impeller sizing tool.\n"
            "Build parameters, run sweeps, review results, and export datasets.",
        )

    def _apply_stylesheet(self) -> None:
        style_path = Path(__file__).resolve().parent / "resources" / "style.qss"
        if style_path.exists():
            with style_path.open("r", encoding="utf-8") as fh:
                self.setStyleSheet(fh.read())


def run_app() -> int:
    app = QApplication.instance()
    owns_app = False
    if app is None:
        app = QApplication(sys.argv)
        owns_app = True
    window = MainWindow()
    window.show()
    if owns_app:
        return app.exec()
    return 0
