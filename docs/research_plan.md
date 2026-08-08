# Research Plan & Methodology

**Thesis Title:** *Adaptive Low-Latency Fraud Detection in Streaming Financial Systems with LLM-Augmented Explainability*  
**Candidate:** Parminder Singh  
**Milestone:** Phase-2 Implementation & Experimentation (Term 2 / 22 Credits)

---

## 1. Research Objectives & Problem Statement

Financial fraud detection in streaming environments suffers from performance degradation due to **concept drift**—changes in customer behavior, fraud patterns, and transaction characteristics over time.

While streaming machine learning algorithms and drift detectors identify when distribution shifts occur, detecting drift does not determine the optimal **model adaptation strategy**.

### Central Research Problem
> **How should model adaptation be triggered and scoped after concept drift in streaming financial fraud detection so that predictive performance can be maintained while controlling latency and computational adaptation cost?**

The thesis systematically investigates **model-update and retraining policies** (**P0 Static**, **P1 Periodic**, **P2 Global Drift-Triggered**, **P3 Segment-Aware Drift-Triggered**), measuring predictive performance jointly with operational costs across diverse drift regimes.

---

## 2. Research Questions & Hypotheses

- **RQ1 (Adaptation Policy Effectiveness):** How does policy choice ($P0–P3$) affect fraud performance and operational cost under concept drift?
- **RQ2 (Drift Regime Interaction):** How does policy effectiveness vary across sudden, gradual, recurring, and localized drift?
- **RQ3 (Localized Adaptation):** Can segment-aware adaptation ($P3$) reduce adaptation overhead while maintaining non-inferior PR-AUC under localized drift?
- **RQ4 (Drift Detector Sensitivity):** To what extent do detector sensitivity differences (ADWIN vs. HDDM) impact policy performance rankings?
- **RQ5 (Operational Performance–Cost Trade-Off):** What quantitative trade-offs exist between PR-AUC, inference latency (p50/p95/p99), adaptation latency, and CPU/RAM consumption?
- **RQ6 (Explainability Under Adaptation):** How do adaptive model updates affect SHAP feature attributions, and can LLMs translate attributions into factual narratives?

---

## 3. Methodology & Base Learner Selection

- **Streaming Replay Protocol:** Prequential (test-then-train) temporal replay ($x_t \rightarrow \hat{y}_t \rightarrow \text{evaluate} \rightarrow y_t \rightarrow \text{learn\_one} \rightarrow \text{policy check} \rightarrow \text{retrain\_window}$).
- **Primary Base Learner:** **Standard Hoeffding Tree Classifier** (River). Captures non-linear decision boundaries incrementally while avoiding internal drift confounds.
- **Secondary Benchmark Learner:** **Adaptive Random Forest (ARF)** (E10).
- **Retraining Engine:** **Windowed Re-instantiation Retraining** (`retrain_window`) on sliding historical memory window $W_{\text{adapt}}$.
- **Segmentation Strategy ($P3$):** **Domain-Defined Categorical Sub-populations** (`ProductCD` in IEEE-CIS; `type` in PaySim).
- **Drift Detectors:** **ADWIN** (Primary) and **HDDM** (Secondary sensitivity check).

---

## 4. Evaluation Metrics & Benchmark Datasets

- **Predictive Metrics:** Primary: **PR-AUC (Precision-Recall Area Under Curve)** / Average Precision. Secondary: ROC-AUC, Recall, Precision, F1, F-beta ($\beta=2$).
- **Operational Metrics:** Inference latency (p50, p95, p99 percentiles), adaptation latency duration, throughput ($\text{tx/sec}$), CPU utilization percentage, memory footprint.
- **Datasets:**
  1. **IEEE-CIS Fraud Detection** (Primary real-world dataset for E1–E8).
  2. **PaySim** (Secondary validation dataset for E9).
  3. **ULB Credit Card Fraud** (Prototype dataset for smoke testing and pipeline validation).

---

## 5. Experiment Matrix (E1–E10)

| Exp ID | Experiment Name | Primary Focus | Core Comparison |
| :--- | :--- | :--- | :--- |
| **E1** | Static vs. Adaptive | Adaptation Benefit | $P0$ Static vs. Adaptive Policies ($P1, P2, P3$) |
| **E2** | Policy Comparison | Operational Trade-Offs | Full comparison across $P0, P1, P2, P3$ |
| **E3** | Drift-Regime Interaction | Regime Sensitivity | $P0–P3$ across Sudden, Gradual, Recurring, Localized Drift |
| **E4** | Global vs. Local Adaptation | Localized Efficiency | $P2$ Global vs. $P3$ Segment-Aware under Localized Drift |
| **E5** | Detector Sensitivity | Robustness Check | ADWIN vs. HDDM triggers for $P2$ and $P3$ |
| **E6** | Operational Trade-Offs | Low-Latency Cost | Multi-dimensional cost profiling (p50/p95/p99 latency, CPU/RAM) |
| **E7** | SHAP Feature Stability | Explanation Shift | SHAP attributions pre- vs. post-adaptation |
| **E8** | LLM Narrative Grounding | Faithfulness | LLM narratives evaluated against raw SHAP vectors |
| **E9** | Cross-Dataset Validation | Generalizability | Core $P0–P3$ experiments repeated on PaySim |
| **E10** | ARF Benchmark Comparison | Learner Isolation | Controlled Base Learner vs. ARF under $P0–P3$ |

---

## 6. Publication Roadmap

- **Paper 1 (Primary Target):** *Adaptive Retraining Policies for Streaming Financial Fraud Detection under Concept Drift* (E1–E3, E5–E6).
- **Paper 2 (Primary Target):** *Segment-Aware Adaptation for Localized Concept Drift in Streaming Fraud Detection* (E4, E6, E9).
- **Paper 3 (Conditional Target):** *Explainability of Adaptive Streaming Fraud Detection via SHAP and LLM Narrative Translation* (E7, E8).
