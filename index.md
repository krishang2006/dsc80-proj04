---
title: Predicting Track Popularity from Audio Features
layout: default
---

# Predicting Track Popularity from Audio Features

**Author:** Krishang
**Course:** DSC 80, Spring 2026 — Final Project

---

## Step 1: Introduction

This project uses the **Spotify Music Tracks** dataset (114,000 tracks across 114 genres) to investigate a single question:

> Can we predict whether a track will be popular on Spotify from its audio features and genre — and more specifically, are some audio features systematically tied to higher popularity, or does popularity reduce to genre and artist alone?

For the focused analysis I selected 5 musically distinct genres — **pop, classical, hip-hop, edm, jazz** — covering very different sonic profiles.

The full notebook with code, plots, and prose lives at [`proj04.ipynb`](./proj04.ipynb).

---

## Step 2: Data Cleaning & EDA

The raw CSV was already clean — only 1 row in the full 114k dataset was missing any text field, and zero rows were missing audio features. After dropping the index column, casting `explicit` to bool, and dropping that 1 row, I restricted to the 5-genre subset (n = 5,000 tracks, 1,000 per genre).

Key EDA findings:

- **Popularity differs dramatically by genre.** Mean popularity: `pop` ≈ 48, `hip-hop` ≈ 51, `edm` ≈ 49, `jazz` ≈ 30, `classical` ≈ 13.
- **Audio features cleanly separate genres.** `classical` is high in `acousticness` and `instrumentalness`, low in `danceability`; `edm` is the opposite; `hip-hop` is high in `speechiness`.
- **Within-genre danceability has only a mild positive trend on popularity** — across-genre level differences dominate.

---

## Step 3: Missingness Assessment

**NMAR (at the row level).** The dataset has almost no explicit `NaN`s (1/114,000 rows). But it is a *scrape* of Spotify, and which tracks exist at all is plausibly **NMAR**: very obscure tracks were under-sampled by the original collector, so a track's absence depends on its own (unobserved) popularity. To downgrade this to MAR we'd need Spotify's full internal catalog with play counts.

**Encoded missingness (the real test).** The audio features carry *disguised* missingness: Spotify returns `tempo = 0`, `danceability = 0`, `time_signature = 0`, `valence = 0` when its audio-analysis pipeline fails. A literal 0 BPM tempo is physically meaningless — these are **sentinels for "analysis failed."** Exactly **157 tracks** share these sentinels, overwhelmingly `sleep`/`ambient` (138 of 157 are `sleep`) — beatless soundscapes.

Treating `tempo == 0` as missing, two permutation tests (10,000 shuffles each) settle the mechanism:

| Dependency tested | observed \|mean diff\| | p-value | verdict |
|---|---|---|---|
| tempo-missingness vs **`energy`** | 0.52 | ≈ 0.000 | **MAR** — strong dependency |
| tempo-missingness vs **`key`** | 0.06 | ≈ 0.83 | independent (MCAR-like) |

So tempo is **MAR**: its missingness depends on the *observed* `energy` (failed-analysis tracks are near-silent), not on the unobserved tempo value itself, and is unrelated to musical key (rhythm detection ⟂ pitch detection). These sentinel rows sit outside the 5-genre modeling subset, so they don't contaminate Steps 6–7.

---

## Step 4: Hypothesis Testing

**Question:** Are `pop` tracks systematically more popular than `classical` tracks, or could the gap be due to chance label assignment?

- **H₀:** Mean Spotify popularity of `pop` tracks equals mean popularity of `classical` tracks.
- **H₁:** Mean popularity of `pop` tracks is strictly greater than `classical`.
- **Test statistic:** $\bar X_\text{pop} - \bar X_\text{classical}$ (one-sided).
- **Method:** 10,000-shuffle permutation test, α = 0.05.

**Result:** Observed difference = **34.52** popularity points (pop=47.58, classical=13.05). Permutation p-value ≈ **0.0000** — no shuffle of the genre label produced a difference even close to the observed one. We **reject H₀**: genre is a real, large-effect predictor of popularity. This justifies including `track_genre` as a feature in Steps 6–7.

---

## Step 5: Framing a Prediction Problem

- **Task:** Binary classification.
- **Target:** `is_popular = (popularity >= 50)`.
- **Features at prediction time:** audio features (danceability, energy, acousticness, instrumentalness, liveness, speechiness, valence, loudness, tempo), `duration_ms`, `explicit`, and `track_genre`. *Excluded* to avoid leakage: `popularity` itself.
- **Primary metric:** F1 (the positive "popular" class is 37.6% of the subset — imbalanced, so accuracy is misleading). Secondary: ROC-AUC.

---

## Step 6: Baseline Model

`sklearn` Pipeline → `ColumnTransformer` (passthrough `danceability`, one-hot `track_genre`) → `LogisticRegression`. Trained on 80% of the 5-genre subset (stratified split, `random_state=42`).

| metric | test value |
|---|---|
| F1 | **0.6464** |
| ROC-AUC | **0.7755** |
| Accuracy | 0.6980 |

This baseline is intentionally minimal — 2 features, no interactions, no regularisation tuning. It sets a floor that Step 7 must beat by ≥ 0.05 F1 to count as a real improvement.

---

## Step 7: Final Model

Over the baseline I added **all 9 audio features** plus `duration_ms` and `explicit`, and engineered two features:

- `energy × danceability` — a multiplicative **interaction** (club/pop "hit" signature that a linear model can't represent on its own);
- `log(duration_ms)` — tames the heavy right-skew from long classical/ambient pieces.

Everything runs inside a single `Pipeline` (StandardScaler on quantitatives, one-hot on `track_genre`) so there's no train/test leakage. The estimator is a **`RandomForestClassifier`** tuned with 5-fold `GridSearchCV` (scored on F1) over `n_estimators ∈ {200, 400}`, `max_depth ∈ {None, 10, 20}`, `min_samples_leaf ∈ {1, 2, 5}`. A tuned `GradientBoostingClassifier` was compared and did not beat it.

| Model | F1 | ROC-AUC | Accuracy |
|---|---|---|---|
| Baseline (LogReg, 2 features) | 0.6464 | 0.7755 | 0.6980 |
| **Final (Random Forest)** | **0.8362** | **0.9249** | **0.8770** |

That's a **+0.19 F1** and **+0.15 ROC-AUC** jump. Feature importances show the model leans on `loudness`, `energy`, `acousticness`, `instrumentalness`, and the engineered interaction — real within-genre audio signal, not just the genre one-hots.

---

## Step 8: Fairness Analysis

**Groups:** explicit vs. non-explicit tracks. **Metric:** precision of the "popular" prediction (a false "popular" flag wastes a promotion slot, so we care whether that error is borne unequally).

- **H₀:** model precision is equal across explicit status (fair).
- **H₁:** precision differs.
- **Test statistic:** \|precision(explicit) − precision(non-explicit)\|; 10,000-shuffle permutation test, α = 0.05.

**Result:** precision was 0.878 (explicit) vs 0.831 (non-explicit), an observed gap of **0.046** with **p ≈ 0.45**. We **fail to reject H₀** — no statistically significant evidence of unfairness w.r.t. explicit content. (Only ~110 test tracks are explicit, so the test has limited power; the conclusion is "no detectable unfairness," not "provably perfectly fair.")

---

## Summary

| Step | Result |
|---|---|
| Hypothesis test | pop vs classical gap = 34.5, p ≈ 0 — genre carries strong signal |
| Missingness | tempo-missingness is **MAR** (depends on `energy`, p ≈ 0; independent of `key`, p ≈ 0.83) |
| Baseline | LogReg, F1 = 0.646 / ROC-AUC = 0.776 |
| **Final model** | Random Forest, **F1 = 0.836 / ROC-AUC = 0.925 / acc = 0.877** |
| Fairness | precision parity holds across `explicit` (gap 0.046, p ≈ 0.45) |

*Full analysis notebook: [`proj04.ipynb`](./proj04.ipynb). DSC 80 Project 4, Spring 2026.*
