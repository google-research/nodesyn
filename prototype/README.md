# 🛡️ NodeSyn: Socially Aligned Synthetic Data & Dynamic AI Evaluation Studio

An end-to-end research workbench for AI safety benchmarking, dynamic taxonomy generation, red-teaming evaluation, automated LLM-as-a-judge autorating, and interactive error analysis.

---

## How to run it on your own machine

**Prerequisite:** Python **3.11 or newer**. Check your version with `python3 --version`. `./run.sh` verifies this automatically and will stop with instructions if only an older Python is found.

**Already cloned the repository before?** Get the latest code first, then skip the clone step below:

```bash
cd nodesyn
git pull
cd prototype
```

Otherwise, clone the repository, then navigate to the `prototype` directory:

```bash
git clone https://github.com/google-research/nodesyn.git
cd nodesyn
cd prototype
```

### Quickstart

To install dependencies, securely configure your API key, and launch the application:

```bash
./run.sh
```

The script will:
1. Detect or create a virtual environment (`./venv`).
2. Install all required dependencies (`requirements.txt`).
3. Prompt you in the terminal to select your AI model providers (**Google Gemini**, **OpenAI GPT**, **Anthropic Claude**, or **All Providers**) and securely enter your API keys (using masked input with zero terminal logging).
4. Launch Streamlit with telemetry disabled and display your local testing URL: `http://localhost:8501`.

---

### Manual Setup (Alternative)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="your-gemini-api-key"
# Optional: configure additional model providers
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
streamlit run streamlit_app.py --browser.gatherUsageStats false
```
