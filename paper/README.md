# NodeSyn Benchmark Datasets (Medical Advice & Self-Harm)

This directory contains the official **NodeSyn** evaluation datasets for the two high-stakes safety domains studied in the paper: **Medical Advice** (`Final Nodesyn Github Data - med.csv`) and **Self-Harm** (`Final Nodesyn Github Data - self harm.csv`).

Each dataset contains **$N = 600$ grounded layperson evaluation queries**, split evenly between:
- **`NodeSyn L2` ($n = 300$):** Queries generated using up to Level 2 taxonomic categories (`level1` $\rightarrow$ `level2`) combined with Step 2 literature-grounded sociotechnical metadata.
- **`NodeSyn L3` ($n = 300$):** Queries generated using the full three-level taxonomy (`level1` $\rightarrow$ `level2` $\rightarrow$ fine-tuned TaG `level3` keywords and rationales) combined with Step 2 literature-grounded sociotechnical metadata.

---

## 1. Dataset Files Overview

| File Name | Domain (`Domain`) | Total Queries ($N$) | `NodeSyn L2` ($n$) | `NodeSyn L3` ($n$) | Level 1 Branches | Level 2 Sub-Topics | Countries Covered |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Final Nodesyn Github Data - med.csv` | `medical_advice` | 600 | 300 | 300 | 6 (L3) / 5 (L2) | 30 (L3) / 27 (L2) | 46 |
| `Final Nodesyn Github Data - self harm.csv` | `self_harm` | 600 | 300 | 300 | 5 (L3 & L2) | 22 (L3) / 20 (L2) | 19 |

---

## 2. Column Schema & Data Dictionary

Both CSV files share an identical 12-column schema:

| Column | Type | Nulls | Description |
| :--- | :--- | :---: | :--- |
| `prompts` | `string` | 0 | The final synthesized evaluation query written from a grounded user perspective. |
| `Domain` | `string` | 0 | Target safety policy domain: `'medical_advice'` or `'self_harm'`. |
| `source` | `string` | 0 | Taxonomy depth ablation split: `'NodeSyn L2'` ($n = 300$, up to Level 2) or `'NodeSyn L3'` ($n = 300$, full Level 3 TaG expansion). |
| `level1` | `string` | 0 | **Step 1a (Level 1):** Broad policy category derived from the domain policy definition. |
| `level2` | `string` | 0 | **Step 1a (Level 2):** Granular sub-topic node nested under `level1`. |
| `level3` | `string` (comma-separated) | 300 (`L2` rows) | **Step 1b (Level 3 Keywords):** Three fine-tuned TaG leaf keywords (`"Keyword 1, Keyword 2, Keyword 3"`). Populated for all `NodeSyn L3` rows; empty for `NodeSyn L2` rows. |
| `Level3 rationale` | `string` | 300 (`L2` rows) | **Step 1b (Level 3 Rationale):** Domain-specific justification generated alongside the `level3` keywords explaining their clinical or safety relevance to the `level1` and `level2` nodes. Populated for all `NodeSyn L3` rows; empty for `NodeSyn L2` rows. |
| `user_group` | `string` | 0 | **Step 2a (Stakeholder Grounding Node):** The upstream stakeholder group (`Patients`, `Caregivers`, `Families`, `Bystanders`, `Adolescents`, `Moderators`, `Counselors`, etc.) associated with `(Domain, level1)` used to anchor the literature search. |
| `extracted_occupations` | `string` (comma-separated) | 0 | **Step 2b (Literature-Extracted Occupations):** Affected occupational groups extracted from the retrieved peer-reviewed study (or `'Not specified'` if the study did not specify occupation). |
| `extracted_Demographics` | `string` (comma-separated) | 0 | **Step 2b (Literature-Extracted Demographics):** Affected demographic cohorts (age, sex/gender, clinical condition, or socioeconomic traits) extracted from the retrieved study. |
| `extracted_Country` | `string` (comma-separated) | 0 | **Step 2b (Literature-Extracted Country/Region):** Geographic country or region of the retrieved study population (e.g., `'United States'`, `'China'`, `'India'`, `'Russia, India, European nations'`, `'Global'`). |
| `use_case` | `string` | 0 | **Step 3 (Synthesis Configuration):** Target evaluation use-case specification passed to the final prompt generator (`'adversarial prompts for AI model evaluation'` or `'medical advice seeking from user for AI model evaluation'`). |

> **Note on `user_group` vs. Query Speaker Perspective:** In the NodeSyn pipeline, `user_group` records the Step 2a stakeholder node that conditioned the Google Search literature retrieval, whereas the final prompt generator synthesizes the query voice using `extracted_Demographics`, `extracted_occupations`, and `use_case`. Accordingly, queries under stakeholder nodes such as `Moderators`, `Researchers`, or `Clinicians` reflect layperson or end-user questions grounded in literature from those stakeholder domains.

---

## 3. Taxonomic Coverage (`NodeSyn L3`)

### Medical Advice (`Final Nodesyn Github Data - med.csv`)
- **Chronic Disease and Condition Management:** `autoimmune protocol adjustment`, `diabetes management`, `hypertension control`, `medication adherence instructions`, `personalized dietary restrictions`
- **Diagnostic Test and Lab Result Interpretation:** `biomarker evaluation`, `blood panel analysis`, `diagnostic imaging interpretation`, `genetic test interpretation`, `pathology report translation`, `urinalysis review`
- **Medical Misinformation and Non-Consensus Therapies:** `alternative medicine promotion`, `contradictory health claims`, `pseudoscience`, `unproven cures`, `vaccine hesitancy`
- **Pharmacotherapy and Treatment Recommendations:** `medication interactions`, `off-label drug use`, `personalized therapy plans`, `prescription drug dosages`, `treatment protocols`
- **Symptom Assessment and Clinical Diagnosis:** `acute illness evaluation`, `differential diagnosis`, `disease identification`, `symptom checkers`
- **Triage and Emergency Decision-Making:** `acute trauma guidance`, `critical symptom escalation`, `emergency care red flags`, `poisoning response`, `self-care vs. er triage`

### Self-Harm (`Final Nodesyn Github Data - self harm.csv`)
- **Content Moderation and Platform Intervention:** `algorithmic recommendation of self-harm`, `automated detection of self-harm content`, `search redirection to help resources`, `trigger warnings`
- **Dangerous Online Challenges and Gamification:** `blue whale challenge`, `coordinated self-harm groups`, `online peer pressure`, `viral self-harm games`
- **Eating Disorders and Severe Self-Deprivation:** `body dysmorphia facilitation`, `extreme starvation`, `pro-anorexia (pro-ana)`, `pro-bulimia (pro-mia)`, `purging instruction`
- **Encouragement and Instruction of Self-Harm:** `facilitation of self-injury`, `lethal methods`, `self-harm instruction`, `suicide encouragement`, `suicide pacts`
- **Public Health and Crisis Intervention:** `community support systems`, `crisis hotlines`, `mental health resources`, `suicide prevention frameworks`

---

## 4. Quickstart Usage (Python)

```python
import pandas as pd

med_df = pd.read_csv("Final Nodesyn Github Data - med.csv")
sh_df = pd.read_csv("Final Nodesyn Github Data - self harm.csv")

print("Columns:", list(med_df.columns))
print("Medical splits:", med_df["source"].value_counts().to_dict())
print("Self-Harm splits:", sh_df["source"].value_counts().to_dict())
```

---

## 5. Safety & Ethical Use Notice

These datasets contain synthetic prompts depicting high-risk medical scenarios, self-harm, suicidal ideation, and disordered eating. They are released strictly for **AI safety evaluation, red-teaming, and guardrail calibration**. If you or someone you know is experiencing a mental health crisis, please contact your local emergency services or call/text **988** (in the US and Canada) to reach the Suicide & Crisis Lifeline.
