import subprocess
import sys


def test_cli_smoke() -> None:
    cmd = [
        sys.executable,
        "-m",
        "baines.cli",
        "--pr",
        "2.5",
        "--beta2-start",
        "0",
        "--beta2-step",
        "10",
        "--nsteps",
        "5",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert any("beta_b2" in line for line in result.stdout.splitlines())
