# 📋 Refined Action Plan v5.0 – Validation and Certification

> This plan does not change the academic content of the guide.
> It adds a layer of **external validation**, **data rigor**, and **admission-style practice** across the same 24 weeks.

**Language:** English | [Español →](../PLAN_V5_ESTRATEGICO.md)

---

## 🎯 Goal of v5.0

- You know you truly master the content (v3.x + v4.0).
- A third party (mentor/AI/interviewer) can confirm your level.
- Your execution matches the admission exam format.

v5.0 introduces 5 protocols on top of the base guide:

1. **Protocol 1 – Data Rigor (Dirty Data Check)**
2. **Protocol 2 – External Validation (Whiteboard Challenge)**
3. **Protocol 3 – Admission Exam Simulation**
4. **Protocol D – Generative Visualization (geometric intuition by code)**
5. **Protocol E – Cognitive Rescue and Execution (metacognition + bridges + PB drills + badges)**

---

## 📦 Relationship with other documents

- Base 24-week plan: [PLAN_ESTUDIOS.md](PLAN_ESTUDIOS.md)
- Daily execution and PyTorch bridge (v4): [PLAN_V4_ESTRATEGICO.md](PLAN_V4_ESTRATEGICO.md)
- Theory exam drills: `study_tools/SIMULACRO_EXAMEN_TEORICO.md`

New v5.0 tools:

- `study_tools/DIRTY_DATA_CHECK.md`
- `study_tools/DESAFIO_TABLERO_BLANCO.md`
- `study_tools/EXAMEN_ADMISION_SIMULADO.md`
- `study_tools/CIERRE_SEMANAL.md`
- `study_tools/DIARIO_METACOGNITIVO.md`
- `study_tools/TEORIA_CODIGO_BRIDGE.md`
- `study_tools/BADGES_CHECKPOINTS.md`
- `study_tools/SIMULACRO_PERFORMANCE_BASED.md`

---

## 1️⃣ Protocol 1 – Data Rigor (Dirty Data Check)

> “Code is useless if the data is garbage.”

### 1.1 Module 01 – Pandas/NumPy (Weeks 1–2)

In addition to loading a CSV and converting to NumPy, the deliverable now includes a **Dirty Data Check**:

- Identify and document at least **5 real issues** in your dataset:
  - missing values / NaN
  - obvious outliers
  - wrong types (strings where numbers should be)
  - strange encodings ("?", "N/A", "-999" as missing)
  - duplicates

For each issue:

- Describe the cleaning strategy (drop, impute, correct, etc.).
- Justify the decision (impact on modeling, sample size, etc.).

Use the template:

- `study_tools/DIRTY_DATA_CHECK.md` (Case 1: Module 01 – initial CSV)

### 1.2 Module 05 – Supervised Learning (Weeks 9–12)

For your first Logistic Regression project:

- Use a real dataset with:
  - categorical variables (requires One-Hot Encoding)
  - numerical variables that need scaling (manual MinMax / StandardScaler)

- Implement a clear preprocessing pipeline **before** the model:
  - cleaning (missing, outliers)
  - categorical encoding
  - scaling
  - train/test split

Document the flow in `DIRTY_DATA_CHECK.md` (Case 2: Module 05 – supervised dataset).

---

## 2️⃣ Protocol 2 – External Validation (Whiteboard Challenge)

> “If you can’t explain it, you don’t understand it.”

v3.2 already mentions the whiteboard challenge; v5.0 formalizes it.

### 2.1 Frequency (Phase 1 and 2)

- 4 mandatory sessions:
  - Week 4
  - Week 8
  - Week 12
  - Week 16

### 2.2 Session format

1. Pick one central concept from recent weeks (e.g., Chain Rule, Gradient Descent, K-Means, PCA, Logistic Regression, Backprop).
2. Prepare a **5-minute explanation** as if speaking to a colleague.
3. Record a short video (screen + voice, camera optional) using:
   - a physical whiteboard,
   - a tablet, or
   - a simple digital board.
4. Request external feedback:
   - mentor, colleague, online community, or
   - an advanced AI (evaluate clarity, correctness, rigor).

Use:

- `study_tools/DESAFIO_TABLERO_BLANCO.md`

### 2.3 Mastery criterion

If in 5 minutes you cannot explain the concept:

- without reading,
- without relying on jargon,
- and without conceptual errors,

then you have not mastered it yet.

---

## 3️⃣ Protocol 3 – Admission exam simulation (Weeks 22 and 23)

> “Train like the real exam is tomorrow.”

This turns Weeks 22–23 into a serious exam training block.

### 3.1 Exam format

- Duration: 2 continuous hours.
- Conditions:
  - no internet
  - no IDE
  - paper + pencil + basic calculator

- Content split:
  - **40% code (pseudocode / steps):** PCA, backprop in a simple network, or K-Means.
  - **60% theory:** loss derivations (e.g., cross-entropy), bias–variance explanation, ML/DL concept questions.

Use:

- `study_tools/EXAMEN_ADMISION_SIMULADO.md`

### 3.2 Calendar

- Week 22 – Simulation 1 (diagnostic)
- Week 23 – Simulation 2 (final)

### 3.3 “Ready for admission” metric

- ≥ 80%: ready
- < 80%: extend 2–4 weeks, reinforce theory, repeat simulation

---

## 4️⃣ Integrated roadmap (v5.0)

| Phase | Content work | v5.0 execution layer |
|---|---|---|
| Weeks 1–8 | Math foundations (Algebra, Calculus, Probability) | Protocol 1: Dirty Data Check in Module 01 |
| Weeks 9–20 | Core ML (Supervised, Unsupervised, Deep Learning) | Protocol 1 + 2: real-data checks + 4 whiteboard sessions |
| Weeks 21–24 | MNIST capstone | Protocol 3: 2 admission simulations |

The PyTorch transition remains detailed in [PLAN_V4_ESTRATEGICO.md](PLAN_V4_ESTRATEGICO.md) and `study_tools/PUENTE_NUMPY_PYTORCH.md`.

---

## 5️⃣ How to use v4.0 and v5.0 together

- v4.0 answers: “How do I study daily and not drop?”
- v5.0 answers: “How do I prove (to myself and others) that I’m ready?”

Use both as layers on top of [PLAN_ESTUDIOS.md](PLAN_ESTUDIOS.md).

---

## 6️⃣ Quick checklist v5.0

- [ ] Module 01: `DIRTY_DATA_CHECK.md` completed for a real CSV.
- [ ] Module 05: Dirty Data Check applied to a supervised dataset with categorical + scaled features.
- [ ] 4 whiteboard videos (Weeks 4, 8, 12, 16) recorded and evaluated.
- [ ] Simulation 1 (Week 22) completed and analyzed.
- [ ] Simulation 2 (Week 23) ≥ 80%.
- [ ] Error diary updated with conceptual issues found in drills/simulations.

---

## 7️⃣ Definitive action protocol v5.0 – A/B/C

### Protocol A – Triple verification (math correctness)

| Module/Week | Validation action | Safety goal |
|---|---|---|
| **Module 03 – Calculus** | Gradient checking via finite differences before trusting any manual `backward()` | Catch sign/algebra bugs |
| **Module 05 – Supervised** | Shadow Mode: compare with `sklearn` on the same dataset | Confirm from-scratch matches industry baseline |
| **Module 07 – Deep Learning** | Overfit test: memorize a tiny batch | Confirm forward/backward logic is correct |

### Protocol B – Academic rigor + external validation

| Frequency | Action | Goal |
|---|---|---|
| Daily (15 min) | Error diary of the day’s biggest conceptual mistakes | Avoid repeating errors; build a personal exam summary |
| Monthly (Saturday) | Whiteboard challenge + external feedback | Prove conceptual mastery like an interview |
| Weeks 22–23 | Admission simulations | Objective “ready / not ready” metric |

### Protocol C – Bridge to reality (industry tooling)

1. After finishing MNIST Analyst with NumPy, rewrite the most advanced network in PyTorch.
2. Use `study_tools/PUENTE_NUMPY_PYTORCH.md` and the DL module mapping.
3. Goal: see how many lines collapse into a few `torch.nn` layers.

 4. Add real-data error analysis (Module 08):

    - Create a section in your final report titled **“5 worst classified images”**.
    - For each image:
      - show the image and the wrong prediction
      - describe the error type briefly
      - argue whether the failure is mostly **bias** (model too simple) or **variance** (too complex / noisy data / insufficient data)

---

## Protocol E – Cognitive rescue and execution

Goal: reduce cognitive fatigue, increase retention, and keep motivation visible.

Components:

1. Weekly cognitive closing (Saturday, 1h): `study_tools/CIERRE_SEMANAL.md`
2. Daily metacognition (5 min): `study_tools/DIARIO_METACOGNITIVO.md`
3. Theory ↔ code bridge (weekly 20–30 min): `study_tools/TEORIA_CODIGO_BRIDGE.md`
4. Badges per module: `study_tools/BADGES_CHECKPOINTS.md`
5. Performance-based drills (Weeks 8, 16, 23): `study_tools/SIMULACRO_PERFORMANCE_BASED.md`

---

## Protocol D – Generative visualization (geometric intuition by code)

Goal: build intuition through **reproducible visualizations** (not static images).

Rule:

- each visualization should answer a conceptual question

Examples:

### Week 3 (Linear Algebra) – Linear transforms and eigenvectors

- **Task:** run and modify `../visualizations/viz_transformations.py`.
- **Deliverable:** a before/after grid plot and a 3–5 line note: “what is the eigenvector (if it exists) and what happens to it?”.

### Week 7 (Calculus) – Interactive 3D gradient descent

- **Task:** run `../visualizations/viz_gradient_3d.py` (Plotly export to HTML).
- **Deliverable:**
  - one run with small `lr` (converges)
  - one run with large `lr` (diverges)
  - a 3–5 line explanation of why.

### Week 19 (CNNs) – Convolution and feature maps (Sobel)

- **Task:** run `../visualizations/viz_convolution.py`.
- **Deliverable:** apply Sobel to a real image and explain what pattern the filter detects.

Required tools:

- `matplotlib`
- `plotly`
- `ipywidgets`
