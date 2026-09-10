# Experiment Execution Log

**Thesis Title:** *Adaptive Low-Latency Fraud Detection in Streaming Financial Systems with LLM-Augmented Explainability*  
**Term Breakdown:** Term 2 (Experiments E1–E5, E9, E10) | Term 3 (Experiments E6, E7, E8)  
**Document Status:** Experiment Registry & Execution Tracker (Aligned with Final Source of Truth)

---

## 1. Experiment Registry & Phasing Matrix

### Term 2 Core Experiments (22 Credits — Streaming Adaptation & Robustness)

| Exp ID | Experiment Name | Primary Target RQ / Obj | Dataset | Policy Matrix | Base Learner | Drift Detector | Status | Output Directory |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **E1** | Static vs. Adaptive | RQ1 / Obj 1 | IEEE-CIS | $P0$ vs. $(P1, P2, P3)$ | Hoeffding Tree | ADWIN | Pending | `results/ieee_cis/E1/` |
| **E2** | Policy Comparison | RQ1 / Obj 1 | IEEE-CIS | $P0, P1, P2, P3$ | Hoeffding Tree | ADWIN | Pending | `results/ieee_cis/E2/` |
| **E3** | Drift-Regime Interaction | RQ2 / Obj 2 | IEEE-CIS | $P0–P3$ across Regimes | Hoeffding Tree | ADWIN | Pending | `results/ieee_cis/E3/` |
| **E4** | Global vs. Local Adaptation | RQ3 / Obj 3 | IEEE-CIS | $P2$ Global vs. $P3$ Segment | Hoeffding Tree | ADWIN | Pending | `results/ieee_cis/E4/` |
| **E5** | Detector Sensitivity | RQ4 / Obj 4 | IEEE-CIS | $P2, P3$ | Hoeffding Tree | ADWIN vs. HDDM | Pending | `results/ieee_cis/E5/` |
| **E9** | Cross-Dataset Validation | RQ1–RQ4 / Generalizability | PaySim | $P0–P3$ | Hoeffding Tree | ADWIN | Pending | `results/paysim/E9/` |
| **E10** | ARF Benchmark Comparison | Robustness / DEC-02 | IEEE-CIS | $P0–P3$ | Adaptive Random Forest | ADWIN | Pending | `results/ieee_cis/E10/` |

### Term 3 Advanced Experiments (44 Credits — Operational Profiling & Explainability)

| Exp ID | Experiment Name | Primary Target RQ / Obj | Dataset | Policy Matrix | Base Learner | Evaluator / Tool | Status | Output Directory |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **E6** | Operational Trade-Offs | RQ5 / Obj 5 | IEEE-CIS | $P0–P3$ | Hoeffding Tree | `psutil` + Latency Profiler | Pending | `results/ieee_cis/E6/` |
| **E7** | SHAP Feature Stability | RQ6 / Obj 6 | IEEE-CIS | $P2, P3$ | Hoeffding Tree | TreeSHAP / KernelSHAP | Pending | `results/ieee_cis/E7/` |
| **E8** | LLM Narrative Grounding | RQ6 / Obj 6 | IEEE-CIS | $P2, P3$ | Hoeffding Tree | LLM Evaluator + Faithfulness Score | Pending | `results/ieee_cis/E8/` |

---

## 2. Logged Experimental Runs

| Run ID | Exp ID | Term | Date | Seed | Dataset | Policy | PR-AUC | p95 Latency (ms) | Adaptation Time (s) | Memory (MB) | Run Config / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| *Run_001* | *E1* | *Term 2* | *YYYY-MM-DD* | *42* | *IEEE-CIS* | *P0* | *TBD* | *TBD* | *N/A* | *TBD* | *Baseline Static Run* |
