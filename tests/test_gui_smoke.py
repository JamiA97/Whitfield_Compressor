import os
import sys

from PySide6.QtWidgets import QApplication

from baines.gui.app import MainWindow


def ensure_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_gui_run_populates_results() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = ensure_app()
    window = MainWindow()
    window.parameter_panel.apply_values(
        {
            "pr": 2.5,
            "beta2_start": 0.0,
            "beta2_step": 10.0,
            "nsteps": 4,
        }
    )
    window.controller.on_run_clicked()
    app.processEvents()
    df = window.results_table.current_dataframe()
    assert len(df) == 4
    window.close()
