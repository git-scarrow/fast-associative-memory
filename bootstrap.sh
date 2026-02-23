#!/usr/bin/env bash
# bootstrap.sh — Cross-platform environment provisioning for Shutter-Deck FAM.
# Supports macOS (Apple Silicon) and Linux (Raspberry Pi 4, Debian/Ubuntu).
set -euo pipefail

VENV_DIR=".venv"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_FILE="shutter-deck-fam.service"

OS="$(uname -s)"
echo "==> Detected OS: ${OS}"

# ── 1. System dependencies ──────────────────────────────────────────────────
echo "==> Installing system dependencies …"
case "${OS}" in
    Darwin)
        if ! command -v brew &>/dev/null; then
            echo "ERROR: Homebrew is required on macOS. Install from https://brew.sh" >&2
            exit 1
        fi
        brew install libraw
        ;;
    Linux)
        sudo apt-get update
        sudo apt-get install -y libraw-dev libopenblas-dev python3-venv
        ;;
    *)
        echo "ERROR: Unsupported OS '${OS}'." >&2
        exit 1
        ;;
esac

# ── 2. Python virtual environment ───────────────────────────────────────────
echo "==> Creating virtual environment at ${VENV_DIR} …"
python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip …"
pip install --upgrade pip

# ── 3. Hardware-aware PyTorch installation ───────────────────────────────────
echo "==> Installing PyTorch …"
case "${OS}" in
    Darwin)
        pip install torch torchvision
        ;;
    Linux)
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        ;;
esac

# ── 4. Application dependencies ─────────────────────────────────────────────
echo "==> Installing application dependencies from requirements.txt …"
pip install -r "${PROJECT_DIR}/requirements.txt"

# ── 5. Systemd service deployment (Linux only) ──────────────────────────────
if [ "${OS}" = "Linux" ]; then
    echo "==> Deploying systemd unit …"

    VENV_PYTHON="${PROJECT_DIR}/${VENV_DIR}/bin/python"
    UNIT_SRC="${PROJECT_DIR}/${SERVICE_FILE}"
    UNIT_DST="/etc/systemd/system/${SERVICE_FILE}"

    # Patch ExecStart to use the absolute venv python path
    sed "s|^ExecStart=.*|ExecStart=${VENV_PYTHON} -m shutter_deck.service ${PROJECT_DIR}/config.toml|" \
        "${UNIT_SRC}" | \
    sed "s|^WorkingDirectory=.*|WorkingDirectory=${PROJECT_DIR}|" | \
    sudo tee "${UNIT_DST}" >/dev/null

    sudo systemctl daemon-reload

    echo ""
    echo "──────────────────────────────────────────────────"
    echo "  Systemd service installed: ${SERVICE_FILE}"
    echo ""
    echo "  Start now:        sudo systemctl start ${SERVICE_FILE}"
    echo "  Enable on boot:   sudo systemctl enable ${SERVICE_FILE}"
    echo "  Check status:     sudo systemctl status ${SERVICE_FILE}"
    echo "──────────────────────────────────────────────────"
fi

echo ""
echo "==> Bootstrap complete. Activate with:  source ${VENV_DIR}/bin/activate"
