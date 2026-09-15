#!/usr/bin/env bash
# ==============================================================================
# Setup Python Virtual Environment, Activate, and Install Requirements
#
# Usage:
#   source scripts/setup_venv.sh       # Sets up, installs, and activates in current shell
#   bash scripts/setup_venv.sh         # Sets up venv and installs dependencies
# ==============================================================================

# Check if script is sourced or executed directly
(return 0 2>/dev/null) && IS_SOURCED=true || IS_SOURCED=false

if [ "$IS_SOURCED" = false ]; then
    set -e
fi

handle_error() {
    local exit_code=$?
    echo "[ERROR] Setup failed with exit code $exit_code." >&2
    if [ "$IS_SOURCED" = true ]; then
        return $exit_code 2>/dev/null || true
    else
        exit $exit_code
    fi
}

# Resolve project root and venv directory
VENV_NAME="${1:-venv}"
if [ -n "$BASH_SOURCE" ]; then
    SCRIPT_PATH="${BASH_SOURCE[0]}"
elif [ -n "$ZSH_VERSION" ]; then
    SCRIPT_PATH="${(%):-%x}"
else
    SCRIPT_PATH="$0"
fi

PROJECT_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
cd "$PROJECT_ROOT" || handle_error

echo "======================================================================"
echo "  Setting up Python Virtual Environment in: $PROJECT_ROOT/$VENV_NAME"
echo "======================================================================"

# 1. Locate Python 3 binary
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "[ERROR] Neither 'python3' nor 'python' was found in PATH." >&2
    handle_error
fi

PY_VERSION=$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
echo "[OK] Detected Python interpreter: Python $PY_VERSION ($PYTHON_BIN)"

# 2. Create virtual environment if it does not exist
if [ ! -d "$VENV_NAME" ] || [ ! -f "$VENV_NAME/bin/activate" ]; then
    echo "[...] Creating virtual environment '$VENV_NAME'..."
    $PYTHON_BIN -m venv "$VENV_NAME" || handle_error
    echo "[OK] Virtual environment created successfully."
else
    echo "[OK] Found existing virtual environment at '$VENV_NAME'."
fi

# 3. Activate the virtual environment
ACTIVATE_PATH="$PROJECT_ROOT/$VENV_NAME/bin/activate"
if [ -f "$ACTIVATE_PATH" ]; then
    # shellcheck source=/dev/null
    source "$ACTIVATE_PATH" || handle_error
    echo "[OK] Virtual environment activated: $(which python)"
else
    echo "[ERROR] Activation script not found at '$ACTIVATE_PATH'" >&2
    handle_error
fi

# 4. Upgrade packaging tools
echo "[...] Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel --quiet || handle_error

# 5. Install requirements.txt
if [ -f "requirements.txt" ]; then
    echo "[...] Installing dependencies from requirements.txt..."
    pip install -r requirements.txt || handle_error
    echo "[OK] Requirements installed successfully."
else
    echo "[WARN] requirements.txt not found in $PROJECT_ROOT."
fi

# 6. Verify core dependencies
echo ""
echo "----------------------------------------------------------------------"
echo "Verifying environment packages..."
python -c "
import numpy as np
import scipy as sp
import cornac
import sklearn
print(f'  - Python Executable : $(which python)')
print(f'  - NumPy             : {np.__version__}')
print(f'  - SciPy             : {sp.__version__}')
print(f'  - Scikit-Learn      : {sklearn.__version__}')
print(f'  - Cornac            : {cornac.__version__}')
" || handle_error
echo "----------------------------------------------------------------------"
echo ""
echo "[DONE] Environment setup complete!"

if [ "$IS_SOURCED" = true ]; then
    echo ">> The virtual environment ($VENV_NAME) is now ACTIVE in your shell."
else
    echo "To activate this environment in your current shell, run:"
    echo "    source $VENV_NAME/bin/activate"
fi
echo "======================================================================"
