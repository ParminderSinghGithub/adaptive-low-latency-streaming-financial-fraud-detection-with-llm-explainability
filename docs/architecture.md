# System Architecture & Design

**Thesis Title:** *Adaptive Low-Latency Fraud Detection in Streaming Financial Systems with LLM-Augmented Explainability*  
**Phase:** Term 2 — 22-Credit Implementation & Experimentation  
**Document Status:** Technical Architecture Specification

---

## 1. High-Level Architecture

The Phase-2 system provides a modular, controlled streaming framework designed to evaluate adaptive retraining policies (**P0–P3**) without implementation confounds.

```text
+-----------------------------------------------------------------------+
|                         EXPERIMENT RUNNER                             |
|  (Loads YAML config, sets seeds, instantiates pipeline, logs metrics) |
+-----------------------------------------------------------------------+
                                   │
                                   ▼
+-----------------------------------------------------------------------+
|                       STREAMING REPLAY ENGINE                         |
|  (Reads dataset, enforces temporal order, streams samples prequentially)|
+-----------------------------------------------------------------------+
                                   │
                                   ▼
+-----------------------------------------------------------------------+
|                      FEATURE PREPROCESSING PIPELINE                   |
|  (Applies leakage-safe encoding, scaling, missing value imputation)   |
+-----------------------------------------------------------------------+
                                   │
                                   ▼
+-----------------------------------------------------------------------+
|                        ADAPTIVE POLICY ENGINE                         |
|  ┌─────────────────────────────────────────────────────────────────┐  |
|  │ Policy Manager (P0 Static, P1 Periodic, P2 Global, P3 Segment)  │  |
|  ├─────────────────────────────────────────────────────────────────┤  |
|  │ Drift Monitor Wrapper (ADWIN / HDDM Global & Segment Instances) │  |
|  ├─────────────────────────────────────────────────────────────────┤  |
|  │ Base Learner Wrapper (Standard Hoeffding Tree / ARF Benchmark) │  |
|  └─────────────────────────────────────────────────────────────────┘  |
+-----------------------------------------------------------------------+
                                   │
                                   ▼
+-----------------------------------------------------------------------+
|                     EVALUATION & METRICS TRACKER                      |
|  (Calculates streaming PR-AUC, logs p50/p95/p99 latency, CPU/RAM)     |
+-----------------------------------------------------------------------+
                                   │
                                   ▼
+-----------------------------------------------------------------------+
|                      EXPLAINABILITY MODULE (SHAP + LLM)               |
|  (Computes SHAP values pre/post adaptation; generates LLM narratives) |
+-----------------------------------------------------------------------+
```

---

## 2. Streaming Data Flow & Replay Engine

- **Replay Engine:** Reads static datasets (IEEE-CIS / PaySim / ULB) and streams transactions sequentially in strict chronological order.
- **Prequential Protocol:**
  1. Transaction feature vector $x_t$ passed to active model.
  2. Inference prediction $\hat{y}_t$ generated; latency logged.
  3. Prediction recorded for metric calculation.
  4. True label $y_t$ revealed.
  5. Incremental online update `learn_one(x_t, y_t)` executed if allowed.
  6. Prediction error $e_t = |y_t - \hat{y}_t|$ sent to drift detector.
  7. Policy Engine evaluates trigger conditions.
  8. If triggered, `retrain_window` executed on memory window $W_{\text{adapt}}$.

---

## 3. Adaptive Retraining & Model Registry

- **Shared Retraining Engine (`retrain_window`):** When policy $P1, P2, \text{ or } P3$ triggers, a fresh instance of the base learner (Standard Hoeffding Tree) is instantiated and trained on the sliding memory window $W_{\text{adapt}}$.
- **Scope Isolation:**
  - *Global Adaptation ($P1, P2$):* Replaces active global model.
  - *Segment-Aware Adaptation ($P3$):* Replaces active model instance dedicated to the specific categorical segment (`ProductCD` / `type`).

---

## 4. Concept Drift Detection Module

- **Primary Detector Wrapper:** **ADWIN** (Adaptive Windowing) dynamically tracking streaming error rates.
- **Secondary Detector Wrapper:** **HDDM** (Hoeffding Drift Detection Method) for detector sensitivity comparison (E5).
- **Segment Detectors ($P3$):** Independent ADWIN instances monitoring sub-stream error rates per categorical transaction segment.

---

## 5. Explainability Layer (SHAP & LLM Translator)

- **SHAP Module:** Computes TreeSHAP / KernelSHAP attribution vectors for sample predictions pre- and post-adaptation. Computes Spearman's rank correlation $\rho$ across retraining boundaries (E7).
- **LLM Narrative Translator:** Receives raw feature values and top SHAP attribution magnitudes. Prompt restricts LLM to factual narrative generation without ungrounded feature hallucination (E8).

---

## 6. Real-Time Latency & Throughput Constraints

- **Inference Latency Tracker:** Measures prediction execution time per sample (logged as p50, p95, p99 percentiles).
- **Adaptation Latency Tracker:** Measures clock time duration required for `retrain_window` execution.
- **Resource Monitor:** Captures CPU utilization percentage and peak RAM footprint (MB/GB) per stream batch.
