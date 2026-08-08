# Experiment Execution Log

**Thesis Title:** *Adaptive Low-Latency Fraud Detection in Streaming Financial Systems with LLM-Augmented Explainability*  
**Phase:** Term 2 — 22-Credit Implementation & Experimentation

---

## Experiment Registry & Execution Tracker

| Exp ID | Experiment Name | Primary Target RQ | Dataset | Policy Matrix | Base Learner | Drift Detector | Status | Output Artifact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **E1** | Static vs. Adaptive | RQ1 / H1 | IEEE-CIS | $P0$ vs. $(P1, P2, P3)$ | Hoeffding Tree | ADWIN | Pending | `results/ieee_cis/E1/` |
| **E2** | Policy Comparison | RQ1 / H1 | IEEE-CIS | $P0, P1, P2, P3$ | Hoeffding Tree | ADWIN | Pending | `results/ieee_cis/E2/` |
| **E3** | Drift-Regime Interaction | RQ2 / H2 | IEEE-CIS | $P0–P3$ across Regimes | Hoeffding Tree | ADWIN | Pending | `results/ieee_cis/E3/` |
| **E4** | Global vs. Local Adaptation | RQ3 / H3 | IEEE-CIS | $P2$ Global vs. $P3$ Segment | Hoeffding Tree | ADWIN | Pending | `results/ieee_cis/E4/` |
| **E5** | Detector Sensitivity | RQ4 / H4 | IEEE-CIS | $P2, P3$ | Hoeffding Tree | ADWIN vs. HDDM | Pending | `results/ieee_cis/E5/` |
| **E6** | Operational Trade-Offs | RQ5 / H5 | IEEE-CIS | $P0–P3$ | Hoeffding Tree | ADWIN | Pending | `results/ieee_cis/E6/` |
| **E7** | SHAP Feature Stability | RQ6 / H6 | IEEE-CIS | $P2, P3$ | Hoeffding Tree | ADWIN | Pending | `results/ieee_cis/E7/` |
| **E8** | LLM Narrative Grounding | RQ6 / H6 | IEEE-CIS | $P2, P3$ | Hoeffding Tree | ADWIN | Pending | `results/ieee_cis/E8/` |
| **E9** | Cross-Dataset Validation | RQ1–RQ5 | PaySim | $P0–P3$ | Hoeffding Tree | ADWIN | Pending | `results/paysim/E9/` |
| **E10** | ARF Benchmark Comparison | Robustness | IEEE-CIS | $P0–P3$ | Adaptive Random Forest | ADWIN | Pending | `results/ieee_cis/E10/` |

---

## Logged Experimental Runs

| Run ID | Exp ID | Date | Seed | Dataset | Policy | PR-AUC | p95 Latency (ms) | Adaptation Time (s) | Memory (MB) | Run Config / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| *Run_001* | *E1* | *YYYY-MM-DD* | *42* | *IEEE-CIS* | *P0* | *TBD* | *TBD* | *N/A* | *TBD* | *Baseline Static Run* |
