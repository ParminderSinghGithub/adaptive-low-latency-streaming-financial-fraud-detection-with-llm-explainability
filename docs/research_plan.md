# Research Plan & Methodology

**Thesis Title:** *Adaptive Low-Latency Fraud Detection in Streaming Financial Systems with LLM-Augmented Explainability*  
**Candidate:** Parminder Singh  
**Term Phasing:** Term 2 (22 Credits: Core Streaming Implementation & Experiments) | Term 3 (44 Credits: Operational Profiling, Explainability & Dissertation)  
**Document Status:** Approved Research Plan (Aligned with Final Source of Truth)

---

## 1. Central Research Problem

Financial fraud detection in streaming environments suffers from performance degradation due to **concept drift**—dynamic shifts in customer behavior, novel fraud tactics, and transaction distributions over time.

While streaming machine learning algorithms and statistical drift detectors identify when distribution shifts occur, detecting drift does not determine the optimal **model adaptation strategy**.

### Central Problem Statement
> **How should model adaptation be triggered and scoped after concept drift in streaming financial fraud detection so that predictive performance can be maintained while controlling latency and computational adaptation cost?**

The thesis systematically investigates **model-update and retraining policies** (**P0 Incremental-only Baseline**, **P1 Periodic**, **P2 Global Drift-Triggered**, **P3 Segment-Aware Drift-Triggered**), measuring predictive performance jointly with operational costs across diverse drift regimes.

---

## 2. Term Phasing & Research Objectives

### Term 2: Core Streaming ML & Adaptation (22 Credits)
- **Objective 1 (RQ1 — Adaptation Policy Effectiveness):** Quantify how policy choice ($P0–P3$) impacts streaming PR-AUC and adaptation overhead under concept drift. (Experiments E1, E2)
- **Objective 2 (RQ2 — Drift Regime Interaction):** Evaluate policy robustness and performance dynamics across sudden, gradual, recurring, and localized drift regimes. (Experiment E3)
- **Objective 3 (RQ3 — Localized Adaptation Efficiency):** Determine whether segment-aware adaptation ($P3$) reduces retraining overhead while maintaining non-inferior PR-AUC under localized drift. (Experiment E4)
- **Objective 4 (RQ4 — Drift Detector Sensitivity):** Assess the sensitivity of policy performance rankings to detector mechanism and threshold calibration (ADWIN vs. HDDM). (Experiment E5)
- **Cross-Dataset & Learner Robustness:** Validate generalizability on PaySim (E9) and benchmark against Adaptive Random Forest (E10).

### Term 3: Advanced Profiling, Explainability & Dissertation (44 Credits)
- **Objective 5 (RQ5 — Operational Performance–Cost Trade-Off):** Construct multi-dimensional Pareto frontiers characterizing empirical trade-offs between streaming PR-AUC, inference latency percentiles (p50/p95/p99), retraining duration, and hardware footprint (CPU %, RAM). (Experiment E6)
- **Objective 6 (RQ6 — Explainability Under Adaptation):** Quantify post-retraining feature attribution stability (TreeSHAP rank correlation) and evaluate hallucination-free LLM narrative synthesis grounded in quantitative SHAP vectors. (Experiments E7, E8)
- **Thesis Completion:** Final dissertation drafting, comprehensive result visualization, and M.Tech thesis defense.

---

## 3. Methodology & Controlled Design Principles

- **Streaming Protocol:** Strict prequential (test-then-train) replay ($x_t \rightarrow \hat{y}_t \rightarrow \text{evaluate} \rightarrow y_t \rightarrow \text{learn\_one} \rightarrow \text{policy check} \rightarrow \text{retrain\_window}$).
- **Primary Base Learner:** **Standard Hoeffding Tree Classifier** (River). Captures incremental non-linear decision boundaries without internal adaptive confounds.
- **Secondary Benchmark Learner:** **Adaptive Random Forest (ARF)** for learner robustness verification (E10).
- **Retraining Mechanism:** **Windowed Re-instantiation Retraining** (`retrain_window`) on sliding historical memory window $W_{\text{adapt}}$.
- **Segmentation Strategy ($P3$):** Domain-defined categorical sub-populations (`ProductCD` in IEEE-CIS; `type` in PaySim).
- **Drift Detection:** **ADWIN** (Primary dynamic windowing) and **HDDM** (Secondary drift sensitivity).

---

## 4. Evaluation Datasets & Metrics

- **Primary Dataset:** **IEEE-CIS Fraud Detection** (590,540 real-world transactions with temporal timestamps; primary for E1–E8, E10).
- **Secondary Dataset:** **PaySim** (Synthetic mobile money transaction stream for generalizability validation in E9).
- **Smoke-Test Dataset:** **ULB Credit Card** (Fast pipeline sanity checking and automated integration testing).
- **Primary Predictive Metric:** **Streaming PR-AUC** (Precision-Recall Area Under Curve / Average Precision).
- **Operational & System Metrics:** Inference latency (p50, p95, p99 percentiles), adaptation execution duration (seconds), CPU utilization percentage, memory footprint (MB).

---

## 5. Experiment Matrix (E1–E10)

| Exp ID | Experiment Name | Term | Primary Target | Core Focus |
| :--- | :--- | :--- | :--- | :--- |
| **E1** | Incremental vs. Adaptive | **Term 2** | RQ1 / Obj 1 | Adaptation benefit: $P0$ Incremental vs. $(P1, P2, P3)$ |
| **E2** | Policy Comparison | **Term 2** | RQ1 / Obj 1 | Full comparison across $P0, P1, P2, P3$ |
| **E3** | Drift-Regime Interaction | **Term 2** | RQ2 / Obj 2 | Robustness across Sudden, Gradual, Recurring, Localized drift |
| **E4** | Global vs. Local Adaptation | **Term 2** | RQ3 / Obj 3 | $P2$ Global vs. $P3$ Segment-Aware under localized drift |
| **E5** | Detector Sensitivity | **Term 2** | RQ4 / Obj 4 | ADWIN vs. HDDM trigger sensitivity comparison |
| **E9** | Cross-Dataset Validation | **Term 2** | Generalizability | Core $P0–P3$ evaluations repeated on PaySim |
| **E10** | ARF Benchmark Comparison | **Term 2** | Robustness | Controlled Hoeffding Tree vs. Adaptive Random Forest |
| **E6** | Operational Trade-Offs | **Term 3** | RQ5 / Obj 5 | Multi-dimensional cost profiling (p50/p95/p99 latency, CPU %, RAM) |
| **E7** | SHAP Feature Stability | **Term 3** | RQ6 / Obj 6 | Feature attribution stability pre- vs. post-retraining |
| **E8** | LLM Narrative Grounding | **Term 3** | RQ6 / Obj 6 | Grounded narrative synthesis without feature hallucination |

---

## 6. Publication Roadmap

- **Paper 1 (Term 2 Target):** *Adaptive Retraining Policies for Streaming Financial Fraud Detection under Concept Drift* (Covering E1–E3, E5; Target: ECML-PKDD, ACM SAC, or IEEE TKDE).
- **Paper 2 (Term 2 Target):** *Segment-Aware Adaptation for Localized Concept Drift in Streaming Fraud Detection* (Covering E4, E9, E10; Target: IEEE BigData, KDD Workshop, or CIKM).
- **Paper 3 (Term 3 Target):** *Explainability of Adaptive Streaming Fraud Detection via SHAP and Grounded LLM Narrative Translation* (Covering E6, E7, E8; Target: ACM ICAIF or FAccT).
