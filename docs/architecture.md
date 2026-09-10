# System Architecture & Design Specification

**Thesis Title:** *Adaptive Low-Latency Fraud Detection in Streaming Financial Systems with LLM-Augmented Explainability*  
**Term Breakdown:** Term 2 (Core Streaming Pipeline & Experiments) | Term 3 (Operational Profiling & Explainability Layer)  
**Document Status:** Technical Architecture Specification (Aligned with Final Source of Truth)

---

## 1. High-Level Architecture Overview

The system architecture is structured in two cohesive phases:
1. **Term 2 Core Streaming & Adaptation Pipeline:** A strictly controlled streaming framework implementing prequential evaluation, drift detection, and the **P0–P3** adaptive retraining policy matrix without implementation confounds.
2. **Term 3 Operational Profiling & Explainability Architecture:** Advanced multi-dimensional system profiling (E6) and post-hoc attribution stability with hallucination-grounded LLM narrative synthesis (E7–E8).

```text
+---------------------------------------------------------------------------------+
|                              EXPERIMENT RUNNER                                  |
|     (Loads YAML config, sets seeds, instantiates pipeline, logs metrics)        |
+---------------------------------------------------------------------------------+
                                         │
                                         ▼
+---------------------------------------------------------------------------------+
|                           STREAMING REPLAY ENGINE                               |
|     (Reads dataset, enforces temporal order, streams samples prequentially)     |
+---------------------------------------------------------------------------------+
                                         │
                                         ▼
+---------------------------------------------------------------------------------+
|                          FEATURE PREPROCESSING PIPELINE                         |
|     (Applies leakage-safe encoding, scaling, missing value imputation)          |
+---------------------------------------------------------------------------------+
                                         │
                                         ▼
+---------------------------------------------------------------------------------+
|                       TERM 2: ADAPTIVE POLICY ENGINE                            |
|  ┌───────────────────────────────────────────────────────────────────────────┐  |
|  │ Policy Manager (P0 Static, P1 Periodic, P2 Global Drift, P3 Segment Drift)│  |
|  ├───────────────────────────────────────────────────────────────────────────┤  |
|  │ Drift Monitor Wrapper (ADWIN / HDDM Global & Segment Instances)           │  |
|  ├───────────────────────────────────────────────────────────────────────────┤  |
|  │ Base Learner Wrapper (Standard Hoeffding Tree / ARF Benchmark)            │  |
|  └───────────────────────────────────────────────────────────────────────────┘  |
+---------------------------------------------------------------------------------+
                                         │
                                         ▼
+---------------------------------------------------------------------------------+
|                     STREAMING EVALUATION & METRICS TRACKER                      |
|     (Streaming PR-AUC, Confusion Matrix, Online Accuracy, Inference Timing)     |
+---------------------------------------------------------------------------------+
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼ (Term 3 Extension)                      ▼ (Term 3 Extension)
+---------------------------------------+  +--------------------------------------+
|  ADVANCED OPERATIONAL PROFILER (E6)   |  |   EXPLAINABILITY MODULE (E7 & E8)    |
|  (p50/p95/p99 Latency, CPU %, RAM)    |  |   (SHAP Stability + LLM Narratives)  |
+---------------------------------------+  +--------------------------------------+
```

---

## 2. Streaming Data Flow & Prequential Protocol (Term 2)

- **Replay Engine:** Reads static datasets (IEEE-CIS / PaySim / ULB) and streams transactions sequentially in strict chronological order ($t$).
- **Prequential (Test-Then-Train) Execution Protocol:**
  1. Transaction feature vector $x_t$ is passed to the active model.
  2. Inference prediction probability $\hat{y}_t$ is generated; sample inference latency is recorded.
  3. Prediction is recorded by the evaluation tracker for streaming PR-AUC computation.
  4. True label $y_t$ is revealed.
  5. Base learner online update `learn_one(x_t, y_t)` is performed (if policy permits incremental learning).
  6. Prediction error metric $e_t = |y_t - \hat{y}_t|$ is passed to the active drift detector(s).
  7. Policy Engine evaluates adaptation trigger conditions ($P0–P3$).
  8. If triggered, `retrain_window` executes on the sliding memory window $W_{\text{adapt}}$.

---

## 3. Adaptive Retraining & Model Registry (Term 2)

- **Shared Retraining Engine (`retrain_window`):** When policy $P1$, $P2$, or $P3$ triggers, a fresh instance of the base learner (Standard Hoeffding Tree) is instantiated and trained on the sliding memory window $W_{\text{adapt}}$.
- **Scope Isolation:**
  - *Global Adaptation ($P1, P2$):* Replaces the global active model.
  - *Segment-Aware Adaptation ($P3$):* Replaces the active model instance dedicated to the specific categorical segment (`ProductCD` in IEEE-CIS; `type` in PaySim).

---

## 4. Concept Drift Detection Module (Term 2)

- **Primary Detector Wrapper:** **ADWIN** (Adaptive Windowing) dynamically tracking streaming error rates.
- **Secondary Detector Wrapper:** **HDDM** (Hoeffding Drift Detection Method) for detector sensitivity comparison (E5).
- **Segment Detectors ($P3$):** Independent ADWIN instances monitoring sub-stream error rates per categorical transaction segment.

---

## 5. Advanced Operational Profiler (Term 3 / Objective 5)

- **High-Resolution Latency Tracker:** Measures per-transaction inference time (logged as p50, p95, p99 percentiles).
- **Adaptation Overhead Monitor:** Measures wall-clock execution time and transaction buffering delays during `retrain_window`.
- **Hardware Footprint Profiler (`psutil`):** Logs sustained and peak CPU utilization percentage and memory footprint across policies ($P0–P3$) for Pareto frontier evaluation (Experiment E6).

---

## 6. Explainability Layer: SHAP & LLM Translator (Term 3 / Objective 6)

- **SHAP Attribution Module:** Computes feature attribution vectors for sample predictions immediately pre- and post-adaptation. Quantifies explanation stability across retraining boundaries via Spearman's rank correlation $\rho$ (Experiment E7).
- **LLM Narrative Translator:** Receives raw feature values and verified top SHAP attributions. Operates under strict prompt grounding constraints to synthesize human-readable fraud alerts without ungrounded feature hallucination (Experiment E8).
