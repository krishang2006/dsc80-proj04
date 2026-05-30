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

The dataset is essentially complete (1/114,000 rows missing). I argue the *hidden* missingness mechanism (which tracks even exist in the API scrape) is plausibly **NMAR**: very obscure tracks were under-sampled by the original collector, so absence from the dataset depends on the missing track's own (unobserved) popularity. Within the dataset itself, missingness is too sparse to support a meaningful permutation test against other columns.

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
- **Primary metric:** F1 (positive class is 10–25% of tracks within each genre — accuracy is misleading). Secondary: ROC-AUC.

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

## Step 7: Final Model (planned)

For final submission I will:

1. Add the remaining 9 audio + metadata features.
2. Engineer two new features: `energy × danceability` interaction term and `log(duration_ms)`.
3. Replace LogisticRegression with tree ensembles (`RandomForestClassifier`, `GradientBoostingClassifier`) and tune `n_estimators`, `max_depth`, `min_samples_leaf` via 5-fold `GridSearchCV` scored on F1.

---

## Step 8: Fairness Analysis (planned)

I will compare F1 on `explicit == True` vs `explicit == False` tracks via a permutation test, with H₀ that model performance is independent of explicit status.

---

*This page is the Checkpoint 2 deliverable for DSC 80 Project 4. The full analysis notebook and code will be linked from this page at final submission.*
