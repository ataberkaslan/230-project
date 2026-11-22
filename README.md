# EE230 Mini-Project: Probabilistic Classification of Heart Disease

## Files
- `heart_disease_probability_project.pdf` — Project instructions and report guidelines.
- `cleveland_heart_data.txt` — Dataset ($297 \times 14$). Cols 1–13 are features; Col 14 is target ($0 \to D=0$, $1\text{--}4 \to D=1$).
- `main.m` — Main MATLAB script that runs the classifiers and prints results.
- `data.md` — Formatted tables and outputs from `main.m` ready to use when writing the report.

## How `main.m` Works
- **Reproducibility:** Fixed seed `rng(77316669)` with a randomized $200 / 97$ train/test split.
- **Output Toggle:** Set `output_on = true` (or `false`) at the top of `main.m` to enable/mute `printf` console output.
- **State Encoding (Part 2):** `get_state` maps the 4 categorical features (`sex`, `cp`, `fbs`, `exang`) into a unique 5-bit state index $x \in \{1, \dots, 32\}$.

## Progress Checklist
- [x] **Part 1 & 2.a:** Data loading, binary target conversion, random train/test split
- [x] **Part 2.b–2.f:** Joint conditional PMF classifier, Laplace smoothing ($\alpha=1$), test accuracy ($79.38\%$) & confusion matrix (logged in `data.md`)
- [ ] **Part 3:** Naive Bayes classifier (same 4 features)
- [ ] **Part 4:** Naive Bayes classifier (all 13 features with tertile discretization)
- [ ] **Part 5:** One-hot encoding, correlation feature selection (top $K$), and logistic regression (`glmfit` / `glmval`)
- [ ] **Parts 2.g, 3.d, 4.d, 5.d:** Repeat experiments with a second random seed/partition
