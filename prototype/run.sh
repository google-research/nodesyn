#!/usr/bin/env bash
# ==============================================================================
# NodeSynth: Socially Aligned Synthetic Data & Dynamic AI Evaluation Studio
# Startup & Dependency Installation Script
# ==============================================================================

set -e

# Ensure running under bash (in case someone ran `sh run.sh`)
if [ -z "$BASH_VERSION" ]; then
    if command -v bash >/dev/null 2>&1; then
        exec bash "$0" "$@"
    else
        echo "❌ Error: Please run this script with bash: bash run.sh"
        exit 1
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "🛡️  NodeSynth Dynamic Evaluation Studio: Setup & Launch"
echo "============================================================"

# ── 1. Python & Virtual Environment Setup ─────────────────────────────────────
PYTHON_BIN=""
for py_cmd in python3.13 python3.12 python3.11 python3 /opt/homebrew/bin/python3 /usr/local/bin/python3 python; do
    if command -v "$py_cmd" >/dev/null 2>&1; then
        resolved_cmd="$(command -v "$py_cmd")"
        # Verify the interpreter actually executes and is not a broken macOS shim
        if "$resolved_cmd" -c "import sys; sys.exit(0)" >/dev/null 2>&1; then
            PYTHON_BIN="$resolved_cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo ""
    echo "❌ Error: No working Python 3 installation was found."
    echo "   • On macOS: run 'xcode-select --install' in terminal, or install Python via Homebrew ('brew install python')."
    echo "   • On Linux: run 'sudo apt install python3 python3-venv' or equivalent."
    echo "   • Or download Python directly from https://www.python.org/downloads/"
    exit 1
fi

echo "[1/4] Using Python: $($PYTHON_BIN --version 2>&1) ($PYTHON_BIN)"

VENV_DIR="$SCRIPT_DIR/venv"
if [ -d "$VENV_DIR" ]; then
    # Test if existing venv is healthy
    if ! "$VENV_DIR/bin/python" -c "import sys; sys.exit(0)" >/dev/null 2>&1; then
        echo "[2/4] Existing venv at ./venv appears invalid. Recreating..."
        rm -rf "$VENV_DIR"
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    else
        echo "[2/4] Using existing virtual environment at ./venv."
    fi
else
    echo "[2/4] Creating virtual environment at ./venv..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# ── 2. Dependency Installation (Cached & Fast) ────────────────────────────────
REQ_STAMP="$VENV_DIR/.requirements_installed"
NEEDS_INSTALL=1

# Check for manual re-install flags
if [ "$1" = "--update" ] || [ "$1" = "--reinstall" ]; then
    NEEDS_INSTALL=1
    shift
else
    if [ -f "$REQ_STAMP" ]; then
        if cmp -s "$SCRIPT_DIR/requirements.txt" "$REQ_STAMP"; then
            if "$VENV_DIR/bin/python" -c "import streamlit, pandas, plotly, google.genai" >/dev/null 2>&1; then
                NEEDS_INSTALL=0
            fi
        fi
    fi
fi

if [ "$NEEDS_INSTALL" -eq 1 ]; then
    echo "[3/4] Installing / updating dependencies from requirements.txt..."
    "$VENV_DIR/bin/pip" install -r requirements.txt
    cp "$SCRIPT_DIR/requirements.txt" "$REQ_STAMP"
    echo "  ↳ Dependencies installed and cached."
else
    echo "[3/4] Dependencies already satisfied. (Skipping pip install for fast startup)"
fi

# ── 3. Gemini API Key Configuration (Secure - No Logging) ────────────────────
echo ""
echo "[4/4] Configuring Gemini API Key..."

export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CURRENT_KEY=""

# Allow key via first command line argument: ./run.sh <key>
if [ -n "$1" ]; then
    CURRENT_KEY="$1"
    echo "  ↳ Using API key passed via argument."
fi

# Check environment variables if not passed via arg
if [ -z "$CURRENT_KEY" ]; then
    CURRENT_KEY="${GEMINI_API_KEY:-$GOOGLE_API_KEY}"
fi

if [ -n "$CURRENT_KEY" ]; then
    echo "  ↳ Detected existing API key in environment."
    read -r -p "    Use existing API key? [Y/n]: " use_existing_choice
    case "$use_existing_choice" in
        [nN][oO]|[nN])
            CURRENT_KEY=""
            ;;
        *)
            echo "    [✓] Continuing with existing key."
            ;;
    esac
fi

if [ -z "$CURRENT_KEY" ]; then
    echo ""
    echo "  👉 Please paste your Gemini API key below and press Enter:"
    echo "     (Note: Text is hidden for security; keystrokes will not be displayed)"
    while [ -z "$CURRENT_KEY" ]; do
        read -s -r -p "     API Key: " entered_key
        echo ""
        entered_key="$(echo "$entered_key" | tr -d '[:space:]')"
        if [ -n "$entered_key" ]; then
            CURRENT_KEY="$entered_key"
            echo "     [✓] API key securely set for this session."
        else
            echo "     ⚠️  Key cannot be empty. Please paste your API key and press Enter:"
        fi
    done
fi

export GEMINI_API_KEY="$CURRENT_KEY"
export GOOGLE_API_KEY="$CURRENT_KEY"
unset CURRENT_KEY
unset entered_key

# ── 4. Launch Streamlit Application ───────────────────────────────────────────
APP_PORT="${PORT:-8501}"
LOCAL_URL="http://localhost:${APP_PORT}"

echo ""
echo "============================================================"
echo "🚀 NodeSynth Dynamic Evaluation Studio is launching!"
echo "📍 Local Web URL: ${LOCAL_URL}"
echo "   Opening browser automatically..."
echo "   (Press Ctrl+C to stop the server)"
echo "============================================================"
echo ""

# Attempt to open web browser in background
(
    sleep 2
    if command -v open >/dev/null 2>&1; then
        open "$LOCAL_URL" >/dev/null 2>&1 || true
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$LOCAL_URL" >/dev/null 2>&1 || true
    fi
) &

exec "$VENV_DIR/bin/streamlit" run streamlit_app.py \
    --server.port "$APP_PORT" \
    --server.headless true \
    --browser.gatherUsageStats false
