# 📋 Improved Action Plan v4.0 - Integration Guide

> This document explains how to integrate the strategic improvements into the existing curriculum plan.

**Language:** English | [Español →](../PLAN_V4_ESTRATEGICO.md)

---

## 🎯 Summary of improvements

This plan does not change **what** you study (the core guide is already strong). It changes **how** you execute so you:

- maintain consistency for 24 weeks
- avoid repeating the same mistakes
- build proof of rigor (typing, linting, tests)

---

## 📁 New resources created

### Environment setup

| File | Description |
|---|---|
| `pyproject.toml` | Python project configuration + dependencies |
| `.pre-commit-config.yaml` | Automated hooks for clean code |
| `setup_env.sh` | Fast install script |

### Study tools (`study_tools/`)

| File | Purpose | When to use |
|---|---|---|
| `DIARIO_ERRORES.md` | Error log | Daily |
| `DRILL_DIMENSIONES_NUMPY.md` | Shape exercises | Weeks 1–2 |
| `SIMULACRO_EXAMEN_TEORICO.md` | Exam-style questions | Saturdays |
| `VISUALIZACION_GRADIENT_DESCENT.md` | Visualization code | Weeks 6–7 |
| `DRYRUN_BACKPROPAGATION.md` | Backprop on paper | Week 18 |
| `PUENTE_NUMPY_PYTORCH.md` | Translation to PyTorch | Week 24 |

### AI code reviewer (`prompts/`)

| File | Description |
|---|---|
| `AI_CODE_REVIEWER.md` | Prompt for using an LLM as a strict code reviewer |

### Tests (`tests/`)

| File | Description |
|---|---|
| `test_dimension_assertions.py` | Shape tests to validate ML code |

---

## 📅 Integration with the existing schedule

### Week 0: Lab setup

Run from the repository root:

```bash
# 1. Setup
bash setup_env.sh

# 2. Activate environment
source venv/bin/activate

# 3. Configure AI Code Reviewer
# Copy prompts/AI_CODE_REVIEWER.md into your preferred LLM tool
```

### Weeks 1–2: Add a daily shape drill

**Daily adjustment (5 min before coding):**

```text
Before coding:
  → Open study_tools/DRILL_DIMENSIONES_NUMPY.md
  → Solve 5 shape-prediction exercises
  → Verify quickly in Python
```

### Weeks 6–7: Add 3D visualization

**Weekly adjustment:**

```text
During Gradient Descent study:
  → Run the code in study_tools/VISUALIZACION_GRADIENT_DESCENT.md
  → Experiment with different learning rates
  → Use a visual tool (e.g., GeoGebra) if helpful
```

### Week 18: Mandatory dry run

**Before implementing Backpropagation:**

```text
1. Open study_tools/DRYRUN_BACKPROPAGATION.md
2. Complete the paper exercise (30 min)
3. Verify with the provided checking code
4. Only then start your implementation
```

### Week 24: Translation to PyTorch

**Extra day at the end of the MNIST capstone:**

```text
1. Open study_tools/PUENTE_NUMPY_PYTORCH.md
2. Take your NumPy NeuralNetwork class
3. Rewrite it in PyTorch (minimal lines)
4. Compare results
5. Answer the “illumination” checklist
```

### Every Saturday: exam drill

**1-hour protocol:**

```text
1. No IDE, no internet
2. Paper + pencil only
3. Open study_tools/SIMULACRO_EXAMEN_TEORICO.md
4. Complete the drill for your current phase
5. Self-score
6. Record weak topics
```

---

## 🔄 Daily “Sandwich” protocol

```text
┌─────────────────────────────────────────────────────────────┐
│ MORNING (Theory - Input)                                    │
│ • Watch/read resources                                      │
│ • Avoid linear note-taking                                  │
│ • Focus on understanding                                    │
├─────────────────────────────────────────────────────────────┤
│ MIDDAY (Implementation - Output)                            │
│ • Write code                                                 │
│ • Use pre-commit as automatic validation                    │
│ • Use AI Code Reviewer for style/vectorization              │
├─────────────────────────────────────────────────────────────┤
│ END (Feynman validation)                                    │
│ • Explain the concept as if teaching someone                │
│ • Log ALL errors in DIARIO_ERRORES.md                       │
│ • Identify: what didn’t I fully understand today?           │
└─────────────────────────────────────────────────────────────┘
```

---

## ✅ Initial setup checklist

- [ ] Run `bash setup_env.sh`
- [ ] Confirm `pre-commit run --all-files` works
- [ ] Configure the AI Code Reviewer prompt
- [ ] Read `study_tools/README.md`
- [ ] Create the first entry in `DIARIO_ERRORES.md`
- [ ] Complete Level 1 in `DRILL_DIMENSIONES_NUMPY.md`

---

## 📊 Adjustment table by phase

| Phase | Weeks | Strategic adjustment | Resource |
|---|---:|---|---|
| Foundations | 1–2 | Shape drill | `DRILL_DIMENSIONES_NUMPY.md` |
| Foundations | 4 | Linear algebra exam drill | `SIMULACRO_EXAMEN_TEORICO.md` |
| Foundations | 6–7 | 3D visualization | `VISUALIZACION_GRADIENT_DESCENT.md` |
| Foundations | 8 | Calculus exam drill | `SIMULACRO_EXAMEN_TEORICO.md` |
| Probability | 12 | Probability exam drill | `SIMULACRO_EXAMEN_TEORICO.md` |
| ML | 16 | Supervised Learning exam drill | `SIMULACRO_EXAMEN_TEORICO.md` |
| DL | 18 | Backprop dry run | `DRYRUN_BACKPROPAGATION.md` |
| DL | 22 | Deep Learning exam drill | `SIMULACRO_EXAMEN_TEORICO.md` |
| Capstone | 24 | PyTorch bridge | `PUENTE_NUMPY_PYTORCH.md` |

---

## 🎯 Success criteria

### Per checkpoint

- Corresponding exam drill ≥ 75 points
- Error log updated
- Code passes pre-commit

### Progress signals

- You can predict `.shape` without running code
- Errors from the diary do not repeat
- Exam drills take < 60 minutes
- You can explain concepts without notes

---

## 🚨 Warning signals

| Signal | Action |
|---|---|
| Same error 3+ times | Relearn the topic from scratch |
| Exam drill < 60 pts | Repeat the phase before moving on |
| Can’t explain without code | More theory, less implementation |
| Pre-commit fails repeatedly | Review style in `AI_CODE_REVIEWER.md` |

---

## 🔗 Quick references

- Main guide: [PLAN_ESTUDIOS.md](PLAN_ESTUDIOS.md)
- Index: [00_INDICE.md](00_INDICE.md)
- Checklist: [CHECKLIST.md](CHECKLIST.md)
- Study tools: `study_tools/README.md`
