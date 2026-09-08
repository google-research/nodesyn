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
            if "$VENV_DIR/bin/python" -c "import streamlit, pandas, plotly, google.genai, openai, anthropic" >/dev/null 2>&1; then
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

# ── 3. AI Model Providers & API Keys Setup ────────────────────────────────────
echo ""
echo "============================================================"
echo "🤖 [4/4] AI Model Providers & API Keys Setup"
echo "============================================================"
echo "NodeSynth supports dynamic generation & evaluation across multiple AI models."
echo "Which model providers are you interested in using?"
echo "  [1] Google Gemini    (Gemini 3.5 Flash, Gemini 3.5 Flash Lite)"
echo "  [2] OpenAI GPT       (GPT-4o, GPT-4o-mini, o3-mini)"
echo "  [3] Anthropic Claude (Claude 3.5 Sonnet, Claude 3.5 Haiku)"
echo "  [4] Meta Llama       (Llama 3.3 70B, Llama 3.1 8B via Groq/OpenRouter)"
echo "  [5] All Providers    (Enable all 4 providers)"
echo ""

export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WANT_GEMINI=0
WANT_OPENAI=0
WANT_CLAUDE=0
WANT_LLAMA=0

# Allow key via first command line argument: ./run.sh <key> (defaults to Gemini key)
if [ -n "$1" ]; then
    export GEMINI_API_KEY="$1"
    export GOOGLE_API_KEY="$1"
    WANT_GEMINI=1
    echo "  ↳ Using Gemini API key passed via argument."
fi

if [ "$WANT_GEMINI" -eq 0 ] && [ "$WANT_OPENAI" -eq 0 ] && [ "$WANT_CLAUDE" -eq 0 ] && [ "$WANT_LLAMA" -eq 0 ]; then
    read -r -p "Enter choice(s) [e.g. 1 or 1,2,3 or all] [Default: 1]: " provider_choice
    provider_choice="${provider_choice:-1}"

    case "$provider_choice" in
        *[aA][lL][lL]*|*5*)
            WANT_GEMINI=1
            WANT_OPENAI=1
            WANT_CLAUDE=1
            WANT_LLAMA=1
            ;;
        *)
            [[ "$provider_choice" =~ 1 ]] && WANT_GEMINI=1
            [[ "$provider_choice" =~ 2 ]] && WANT_OPENAI=1
            [[ "$provider_choice" =~ 3 ]] && WANT_CLAUDE=1
            [[ "$provider_choice" =~ 4 ]] && WANT_LLAMA=1
            ;;
    esac

    # Default to Gemini if no number matched
    if [ "$WANT_GEMINI" -eq 0 ] && [ "$WANT_OPENAI" -eq 0 ] && [ "$WANT_CLAUDE" -eq 0 ] && [ "$WANT_LLAMA" -eq 0 ]; then
        WANT_GEMINI=1
    fi
fi

# ── Gemini Configuration ──────────────────────────────────────────────────────
if [ "$WANT_GEMINI" -eq 1 ]; then
    echo ""
    echo "🔹 Google Gemini Configuration:"
    CURRENT_GEMINI="${GEMINI_API_KEY:-$GOOGLE_API_KEY}"
    if [ -n "$CURRENT_GEMINI" ]; then
        echo "  ↳ Detected existing Gemini API key in environment."
        read -r -p "    Use existing Gemini key? [Y/n]: " use_existing_gemini
        case "$use_existing_gemini" in
            [nN][oO]|[nN]) CURRENT_GEMINI="" ;;
            *) echo "    [✓] Using existing Gemini key." ;;
        esac
    fi
    if [ -z "$CURRENT_GEMINI" ]; then
        echo "    Please paste your Gemini API key (hidden):"
        read -s -r -p "    Gemini API Key: " entered_gemini
        echo ""
        entered_gemini="$(echo "$entered_gemini" | tr -d '[:space:]')"
        if [ -n "$entered_gemini" ]; then
            CURRENT_GEMINI="$entered_gemini"
            echo "    [✓] Gemini API key set for this session."
        else
            echo "    ↳ Skipped for now. (You can also add it in the webapp UI)"
        fi
    fi
    if [ -n "$CURRENT_GEMINI" ]; then
        export GEMINI_API_KEY="$CURRENT_GEMINI"
        export GOOGLE_API_KEY="$CURRENT_GEMINI"
    fi
fi

# ── OpenAI Configuration ──────────────────────────────────────────────────────
if [ "$WANT_OPENAI" -eq 1 ]; then
    echo ""
    echo "🔹 OpenAI (GPT) Configuration:"
    CURRENT_OPENAI="$OPENAI_API_KEY"
    if [ -n "$CURRENT_OPENAI" ]; then
        echo "  ↳ Detected existing OpenAI API key in environment."
        read -r -p "    Use existing OpenAI key? [Y/n]: " use_existing_openai
        case "$use_existing_openai" in
            [nN][oO]|[nN]) CURRENT_OPENAI="" ;;
            *) echo "    [✓] Using existing OpenAI key." ;;
        esac
    fi
    if [ -z "$CURRENT_OPENAI" ]; then
        echo "    Please paste your OpenAI API key (hidden):"
        read -s -r -p "    OpenAI API Key: " entered_openai
        echo ""
        entered_openai="$(echo "$entered_openai" | tr -d '[:space:]')"
        if [ -n "$entered_openai" ]; then
            CURRENT_OPENAI="$entered_openai"
            echo "    [✓] OpenAI API key set for this session."
        else
            echo "    ↳ Skipped for now. (You can also add it in the webapp UI)"
        fi
    fi
    if [ -n "$CURRENT_OPENAI" ]; then
        export OPENAI_API_KEY="$CURRENT_OPENAI"
    fi
fi

# ── Anthropic Configuration ───────────────────────────────────────────────────
if [ "$WANT_CLAUDE" -eq 1 ]; then
    echo ""
    echo "🔹 Anthropic (Claude) Configuration:"
    CURRENT_ANTHROPIC="$ANTHROPIC_API_KEY"
    if [ -n "$CURRENT_ANTHROPIC" ]; then
        echo "  ↳ Detected existing Anthropic API key in environment."
        read -r -p "    Use existing Anthropic key? [Y/n]: " use_existing_anthropic
        case "$use_existing_anthropic" in
            [nN][oO]|[nN]) CURRENT_ANTHROPIC="" ;;
            *) echo "    [✓] Using existing Anthropic key." ;;
        esac
    fi
    if [ -z "$CURRENT_ANTHROPIC" ]; then
        echo "    Please paste your Anthropic API key (hidden):"
        read -s -r -p "    Anthropic API Key: " entered_anthropic
        echo ""
        entered_anthropic="$(echo "$entered_anthropic" | tr -d '[:space:]')"
        if [ -n "$entered_anthropic" ]; then
            CURRENT_ANTHROPIC="$entered_anthropic"
            echo "    [✓] Anthropic API key set for this session."
        else
            echo "    ↳ Skipped for now. (You can also add it in the webapp UI)"
        fi
    fi
    if [ -n "$CURRENT_ANTHROPIC" ]; then
        export ANTHROPIC_API_KEY="$CURRENT_ANTHROPIC"
    fi
fi

# ── Meta Llama Configuration ──────────────────────────────────────────────────
if [ "$WANT_LLAMA" -eq 1 ]; then
    echo ""
    echo "🔹 Meta Llama Configuration:"
    echo "  Select Llama hosting provider:"
    echo "    [1] Groq (Recommended - ultra-fast inference, free tier at console.groq.com)"
    echo "    [2] OpenRouter (openrouter.ai)"
    echo "    [3] Together AI (together.ai)"
    echo "    [4] Custom OpenAI-compatible endpoint (e.g. vLLM / Ollama)"
    read -r -p "  Llama Provider [Default: 1]: " llama_prov_choice
    llama_prov_choice="${llama_prov_choice:-1}"

    case "$llama_prov_choice" in
        2)
            export LLAMA_PROVIDER="openrouter"
            CURRENT_LLAMA="${OPENROUTER_API_KEY:-$LLAMA_API_KEY}"
            KEY_NAME="OpenRouter"
            ;;
        3)
            export LLAMA_PROVIDER="together"
            CURRENT_LLAMA="${TOGETHER_API_KEY:-$LLAMA_API_KEY}"
            KEY_NAME="Together AI"
            ;;
        4)
            export LLAMA_PROVIDER="custom"
            read -r -p "  Enter Custom Endpoint Base URL [Default: http://localhost:11434/v1]: " custom_url
            export LLAMA_BASE_URL="${custom_url:-http://localhost:11434/v1}"
            CURRENT_LLAMA="$LLAMA_API_KEY"
            KEY_NAME="Custom Endpoint"
            ;;
        *)
            export LLAMA_PROVIDER="groq"
            CURRENT_LLAMA="${GROQ_API_KEY:-$LLAMA_API_KEY}"
            KEY_NAME="Groq"
            ;;
    esac

    if [ -n "$CURRENT_LLAMA" ]; then
        echo "  ↳ Detected existing $KEY_NAME key in environment."
        read -r -p "    Use existing $KEY_NAME key? [Y/n]: " use_existing_llama
        case "$use_existing_llama" in
            [nN][oO]|[nN]) CURRENT_LLAMA="" ;;
            *) echo "    [✓] Using existing $KEY_NAME key." ;;
        esac
    fi
    if [ -z "$CURRENT_LLAMA" ]; then
        echo "    Please paste your $KEY_NAME API key (hidden):"
        read -s -r -p "    $KEY_NAME API Key: " entered_llama
        echo ""
        entered_llama="$(echo "$entered_llama" | tr -d '[:space:]')"
        if [ -n "$entered_llama" ]; then
            CURRENT_LLAMA="$entered_llama"
            echo "    [✓] $KEY_NAME API key set for this session."
        else
            echo "    ↳ Skipped for now. (You can also add it in the webapp UI)"
        fi
    fi
    if [ -n "$CURRENT_LLAMA" ]; then
        if [ "$LLAMA_PROVIDER" = "groq" ]; then
            export GROQ_API_KEY="$CURRENT_LLAMA"
        elif [ "$LLAMA_PROVIDER" = "openrouter" ]; then
            export OPENROUTER_API_KEY="$CURRENT_LLAMA"
        elif [ "$LLAMA_PROVIDER" = "together" ]; then
            export TOGETHER_API_KEY="$CURRENT_LLAMA"
        fi
        export LLAMA_API_KEY="$CURRENT_LLAMA"
    fi
fi

# ── Summary of Active Providers ───────────────────────────────────────────────
echo ""
echo "============================================================"
echo "📋 Configured AI Model Providers:"
if [ -n "${GEMINI_API_KEY:-$GOOGLE_API_KEY}" ]; then
    echo "  • Google Gemini:      ✅ Ready"
else
    echo "  • Google Gemini:      ⚪ Not configured"
fi
if [ -n "$OPENAI_API_KEY" ]; then
    echo "  • OpenAI (GPT):       ✅ Ready"
else
    echo "  • OpenAI (GPT):       ⚪ Not configured"
fi
if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "  • Anthropic (Claude): ✅ Ready"
else
    echo "  • Anthropic (Claude): ⚪ Not configured"
fi
if [ -n "${GROQ_API_KEY:-${OPENROUTER_API_KEY:-${TOGETHER_API_KEY:-$LLAMA_API_KEY}}}" ]; then
    echo "  • Meta Llama:         ✅ Ready (${LLAMA_PROVIDER:-groq})"
else
    echo "  • Meta Llama:         ⚪ Not configured"
fi
echo "============================================================"

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
