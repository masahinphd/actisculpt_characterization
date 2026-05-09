from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from venv import EnvBuilder

PROJECT_DIR = Path(__file__).resolve().parent
VENV_DIR = PROJECT_DIR / ".venv"
REQUIREMENTS_FILE = PROJECT_DIR / "requirements.txt"
GUI_FILE = PROJECT_DIR / "GUI.py"


def ensure_virtual_environment() -> Path:
    if not VENV_DIR.exists() or not (VENV_DIR / "pyvenv.cfg").exists():
        print("Creating virtual environment...")
        EnvBuilder(with_pip=True).create(VENV_DIR)
    return get_venv_python()


def get_venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def ensure_requirements(venv_python: Path) -> None:
    if not REQUIREMENTS_FILE.exists():
        return

    print("Upgrading pip...")
    subprocess.check_call([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])

    print("Installing requirements...")
    subprocess.check_call([str(venv_python), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)])


def run_streamlit(venv_python: Path) -> int:
    if not GUI_FILE.exists():
        print(f"Cannot find {GUI_FILE.name} in {PROJECT_DIR}")
        return 1

    cmd = [str(venv_python), "-m", "streamlit", "run", str(GUI_FILE)]
    return subprocess.call(cmd, cwd=str(PROJECT_DIR))


def main() -> int:
    try:
        venv_python = ensure_virtual_environment()
        if not venv_python.exists():
            print(f"Virtual environment Python not found: {venv_python}")
            return 1

        ensure_requirements(venv_python)
        return run_streamlit(venv_python)
    except subprocess.CalledProcessError as exc:
        return exc.returncode or 1
    except Exception as exc:
        print(f"Launcher failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
