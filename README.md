# NodeSyn: Socially Aligned Synthetic Data for AI Evaluation

🚀 [**Launch Live Prototype**](https://nodesyn.streamlit.app/)

**NodeSyn** is a research prototype that implements a scalable, multi-stage methodology for creating socially relevant and evidence-grounded synthetic data (e.g., annotated queries) for AI model evaluation. 

The pipeline breaks down topics related to safety policies (e.g., harassment) and sensitive domains (e.g., education) into granular taxonomies using a fine-tuned taxonomy generator. It identifies key relationships within these taxonomies (e.g., affected social groups, geographic regions, use cases) and generates high-fidelity synthetic queries designed for rigorous model evaluation.

## 🚨 The Challenge
Standard benchmarks and manual query creation struggle to capture real-world sociotechnical nuance or scale effectively. While generic synthetic data offers an alternative, these datasets often contain unintended biases, lack diversity, and are inaccurate for highly-sensitive domains. NodeSyn enables users (researchers, developers, and auditors) to go from a high-level topic to a rich synthetic dataset capturing relationships that represent *documented harms* in the real world.

## 💡 Core Contributions
* **Sociotechnical Framework:** Leverages an expert-curated Taxonomy Generator (TaG) to ground abstract concepts in concrete, evidence-based scenarios.
* **Empowered Scaling:** Enables resource-constrained entities to conduct lightweight, scalable model evaluations specific to sensitive topics.
* **Interpretable Diagnostics:** Allows evaluators to trace exact failure intersections (demographics, geography, taxonomy level) to prioritize key areas of concern for targeted mitigation and in-depth human evaluation.

<p align="center">
  <img src="paper/NodeSyn_flow.svg" alt="NodeSyn Flowchart" width="100%">
  <br>
  <em><b>Figure 1:</b> A visual representation of the NodeSyn approach. Based on user inputs, NodeSyn (Step 1) creates a complete, three layer taxonomy using a fine-tuned model; and (Step 2) extracts metadata (e.g., sensitive characteristics) from relevant sources, related to the branches of the taxonomy. Utilizing the aforementioned concepts and annotations, NodeSyn (Step 3) generates annotated synthetic queries for model evaluation.</em>
</p>


## 🔄 End-to-End Workflow

You can explore the full workflow directly in our [**Live Prototype**](https://nodesyn.streamlit.app/):

1. **Concept Setup:** Define the overarching theme (e.g., "Cultural Bias", "Medical Advice") and operational constraints (countries, languages, modality).
2. **Taxonomy Generation:** The system leverages a fine-tuned taxonomy generator (TaG) to intelligently extrapolate a structured vocabulary (L1, L2, L3).
3. **Data Synthesis:** Generate synthetic examples, anchoring them in intersections of sensitive attributes and complex societal contexts.
4. **Evaluation:** Define a rubric targeting nuanced harms. Evaluate the target model's performance on the synthetic dataset.
5. **Analysis Dashboard:** Perform root-cause analysis. Trace where model performance degrades across taxonomic and demographic intersections.

This prototype and the approach outlined in the [accompanying paper](https://arxiv.org/abs/2605.14381) can be used to conduct lightweight model evaluations specific to sensitive topics, enabling model developers and deployers to prioritize key areas of concern for in-depth human evaluation.

## 📖 Citation

If you use NodeSyn in your research, please cite the following paper:

```bibtex
@article{rashid2026NodeSyn,
  title={NodeSyn: Socially Aligned Synthetic Data for AI Evaluation},
  author={Rashid, Qazi Mamunur and Yang, Xuan and Yang, Zhengzhe and Pan, Yanzhou and van Liemt, Erin and Neal, Darlene and Pancholi, Kshitij and Smith-Loud, Jamila},
  journal={arXiv preprint arXiv:2605.14381},
  year={2026}
}
```
## ⚠️ Disclaimer

This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).

This project is intended for demonstration purposes only. It is not
intended for use in a production environment.

## ⚖️ License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## How to run it on your own machine

Clone the repository, then navigate to the `prototype` directory:

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
