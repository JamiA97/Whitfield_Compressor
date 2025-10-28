# Baines Compressor Sizer

Desktop GUI and CLI around the Baines centrifugal compressor impeller sizing equations. Engineers can sweep discharge blade angles, review calculated performance metrics, visualise trends, and export datasets for downstream use.

![Main window](docs/screenshots/main_window.png)
![Plot panel](docs/screenshots/plots.png)

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
pip install -U pip
pip install -r requirements.txt
```

### CLI

```bash
python -m baines.cli --pr 2.5 --beta2-start 0 --beta2-step 10 --nsteps 5
```

### GUI

```bash
python baines_gui.py
```

## UI Overview

- **Parameter Dock**: enter design pressure ratio, beta sweep parameters, model coefficients, and optional sizing inputs. Built-in presets and user preset actions live here.
- **Results Table**: sortable table backed by pandas, copy with `Ctrl+C`, or right-click to export selected rows to CSV. Hover rows (when sizing inputs provided) to see geometry tooltips.
- **Plots Panel**: tabbed Matplotlib canvases with zoom/pan/save toolbar for default trends vs. blade angle.
- **Status Bar**: reports runtime, row count, and warning count after each run.

## Input Fields

| Field | Units | Default | Notes |
| --- | --- | --- | --- |
| PR | – | 2.5 | Stage total pressure ratio |
| beta2_start | deg | 0 | Starting discharge blade metal angle |
| beta2_step | deg | 10 | Positive decrement per sweep step |
| nsteps | – | 5 | Number of sweep points |
| SF | – | 0.85 | Slip factor |
| ETI | – | 0.90 | Impeller total-to-total efficiency |
| ETS | – | 0.80 | Stage total-to-total efficiency |
| BETA1S | deg | -60 | Inlet shroud relative flow angle |
| ANU | – | 0.40 | Hub-to-shroud radius ratio at inducer |
| ALP2 | deg | 65 | Discharge absolute flow angle |
| R1SR2 | – | 0.70 | Inducer shroud/discharge tip radius ratio |
| gamma | – | 1.40 | Ratio of specific heats |
| mf | kg/s | – | Mass flow rate (optional) |
| p01 | bar | – | Inlet stagnation pressure (optional) |
| t01 | K | – | Inlet stagnation temperature (optional) |

Invalid or incomplete inputs highlight in red; runs are blocked until issues are resolved.

## Presets

Built-in YAML presets live under `baines/gui/presets/`:

- `hd_diesel.yml`
- `pcar.yml`
- `micro_gt.yml`

Save and load user presets from the GUI. Files land under:

- Linux: `~/.config/baines_gui/presets/`
- macOS: `~/Library/Application Support/baines_gui/presets/`
- Windows: `%APPDATA%/baines_gui/presets/`

## Exports & Sessions

- **CSV**: full table or selected rows, via File ▸ Export or table context menu.
- **JSON**: includes header with ISO timestamp and full input set plus ResultRow payloads.
- **Session (.baines.yml)**: saves inputs, results, and row tooltips for later reload.

Warnings flag rows where `M2_rel > 1.2`, `b2/r2` is outside `[0.03, 0.12]`, or `U2/A0`, `PHI`, `PSI` contain NaNs.

## Development

```bash
black .
ruff .
mypy baines/
pytest
```

Runtime packages are defined in `pyproject.toml`; optional `dev` extras install tooling.

## Screenshots

Replace the placeholder PNGs in `docs/screenshots/` with real captures when available.
