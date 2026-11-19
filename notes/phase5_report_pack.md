# Phase 5 Report Fact Pack

Compiled from `results/ml_metrics.txt`, `results/ml/*.csv`, `main/ml/*.scala`, `CLAUDE.md`,
and `notes/phase5_report_material.md` on 2026-11-19, as a single reference for writing the
Phase 5 section of the report. All numbers below are copied directly from the generated
outputs (not re-derived) -- see the source file named at the top of each section if a
number needs to be re-checked.

---

## 1. Split

Source: `results/ml_metrics.txt` lines 8-34, `results/ml/split.csv`.

Track-level 70/30 split, stratified by genre, seed 42. 5-fold grouped CV on training
recordings only. **688 training recordings / 296 test recordings.**

| Genre | Train recordings | Train windows | Test recordings | Test windows |
|---|---|---|---|---|
| blues | 70 | 4130 | 30 | 1770 |
| classical | 70 | 4117 | 30 | 1766 |
| country | 70 | 4130 | 30 | 1767 |
| disco | 69 | 4071 | 30 | 1769 |
| hiphop | 69 | 4073 | 29 | 1713 |
| jazz | 69 | 4071 | 30 | 1770 |
| metal | 64 | 3776 | 28 | 1652 |
| pop | 69 | 4071 | 29 | 1711 |
| reggae | 69 | 4053 | 30 | 1770 |
| rock | 69 | 4070 | 30 | 1770 |
| **Total** | **688** | **40,562** | **296** | **17,458** |

(40,562 + 17,458 = 58,020 -- matches the full dataset. Metal has the fewest train
recordings, 64, because it lost the most windows/tracks to cleaning.)

Disjointness check passed (no `track_id` in both sides). Gap pairs excluded from the
change features (predecessor window removed by cleaning, so the LAG pair isn't truly
consecutive): **train = 4, test = 1** (Phase 4 found 5 across the full 58,020-window
dataset -- 4+1=5, consistent).

**Code (`main/ml/Split.scala`, lines 51-57, the seeded split core):**
```scala
val rng = new scala.util.Random(seed)
val splitRows: Seq[(String, String, String)] = byGenre.flatMap { case (genre, sortedTrackIds) =>
  val shuffled = rng.shuffle(sortedTrackIds)
  val numTrain = math.round(trainFrac * sortedTrackIds.size).toInt
  val (trainIds, testIds) = shuffled.splitAt(numTrain)
  trainIds.map(id => (id, genre, "train")) ++ testIds.map(id => (id, genre, "test"))
}
```

**Code (`main/ml/Split.scala`, lines 82-86, grouped fold assignment):**
```scala
val rng = new scala.util.Random(seed)
val foldRows: Seq[(String, String, Int)] = byGenre.flatMap { case (genre, sortedTrackIds) =>
  val shuffled = rng.shuffle(sortedTrackIds)
  shuffled.zipWithIndex.map { case (trackId, i) => (trackId, genre, i % numFolds) }
}
```

---

## 2. Features

### 2.1 IMPORTANT caveat on the feature set's provenance (read before using these numbers)

`MachineLearning.scala` (line 48) reads `data/processed/final` (`Paths.Final`) as the
base table for every modelling step. That table is Phase 2's `Reduction` + `Transformation`
output: which 37 of the 38 window features survive (Reduction drops `spectral_centroid_mean`)
and which features get a `log_` prefix (Transformation's skewness-based rule) were **both
decided using the full 984-recording / 58,020-window dataset, before Phase 5's train/test
split existed.** `CLAUDE.md`'s own Phase 5 rules section says explicitly: *"Do NOT start
feature selection from data/processed/final... Start from data/interim/02_integrated
instead... recompute eta-squared, pairwise correlations, and skewness using the training
tracks only."* The code only re-derives eta²/correlation on training rows
(`FeatureAnalysis.analyzeTrainFeatures`, operating on `trainDf` which is already carved
from `data/processed/final`) -- it never redoes the skewness-based log-transform choice or
the original Reduction drop from `data/interim/02_integrated` on training tracks only. The
one exception is the narrow spectral_centroid_mean-vs-rolloff_mean recheck
(`FeatureAnalysis.recheckPhase2Selection`), which correctly reads `Paths.Integrated`.

Net effect: eta² and the |r|>0.95 redundancy check below ARE computed on training rows
only (no test row ever enters those formulas), but the *feature space* they're computed
over (37 columns, log-transformed or not) was shaped by statistics that included the
test recordings. This is a real, if likely small, leakage path, and a documented
deviation from `CLAUDE.md`. See Part 3 of the audit for the proposed fix.

### 2.2 F0 base feature set (37 features)

Train-only eta² analysis agrees with Phase 2: **nothing dropped beyond
`spectral_centroid_mean`** (already removed by Phase 2). Top 10 by eta² (train-only,
`results/ml/train_relevance.csv`):

| Rank | Feature | eta² | Decision |
|---|---|---|---|
| 1 | log_flatness_var | 0.4939 | kept |
| 2 | log_rms_var | 0.4878 | kept |
| 3 | mfcc1_mean | 0.4761 | kept |
| 4 | log_flatness_mean | 0.4748 | kept |
| 5 | spectral_bandwidth_mean | 0.4562 | kept |
| 6 | log_spectral_centroid_var | 0.4535 | kept |
| 7 | rolloff_mean | 0.4217 | kept |
| 8 | log_rolloff_var | 0.3804 | kept |
| 9 | rms_mean | 0.3788 | kept |
| 10 | mfcc2_mean | 0.3731 | kept |

`loudness_variation` (the Phase 2 feedback item): eta² = **0.1525**, kept -- row 33 of
`train_relevance.csv`.

### 2.3 Candidate extra features (eta², train-only; not part of F0)

| Feature | eta² |
|---|---|
| range_p95_p5 | 0.419 (see `train_relevance.csv`, "candidate extra feature" rows) |
| d_loudness_mean_abs | 0.292 |
| d_mfcc1_mean_abs | 0.198 |
| d_bandwidth_mean_abs | 0.145 |

(Exact 4-decimal values are in `results/ml/train_relevance.csv`'s trailing 4 rows.)

### 2.4 Pairwise correlations with |r| > 0.8 (train-only), `results/ml/train_correlation_pairs.csv`

12 pairs reported, none newly actioned (action threshold is |r| > 0.95):

| Feature A | Feature B | r |
|---|---|---|
| spectral_bandwidth_mean | rolloff_mean | 0.9467 |
| log_flatness_mean | log_flatness_var | 0.9310 |
| rolloff_mean | mfcc2_mean | -0.9245 |
| rolloff_mean | log_flatness_mean | 0.8894 |
| log_flatness_mean | mfcc2_mean | -0.8882 |
| log_spectral_centroid_var | log_rolloff_var | 0.8813 |
| spectral_bandwidth_mean | log_flatness_mean | 0.8799 |
| log_spectral_bandwidth_var | log_rolloff_var | 0.8777 |
| spectral_bandwidth_mean | mfcc2_mean | -0.8727 |
| spectral_bandwidth_mean | log_flatness_var | 0.8355 |
| rolloff_mean | log_flatness_var | 0.8164 |
| log_zcr_var | log_spectral_centroid_var | 0.8006 |

### 2.5 Phase 2 selection recheck (`results/ml/phase2_selection_recheck.txt`)

Input: `data/interim/02_integrated`, filtered to the 688 training recordings (40,562
window rows -- matches the split table above, cross-checked).

| Quantity | Value |
|---|---|
| eta²(spectral_centroid_mean) | 0.3743 |
| eta²(rolloff_mean) | 0.4217 |
| Pearson r(spectral_centroid_mean, rolloff_mean) | 0.9674 |

Train-only: |r| > 0.95? **true**. spectral_centroid_mean has the lower eta²? **true**.
**Agreement: Phase 2's removal of spectral_centroid_mean still holds under train-only
statistics.** (This recheck is correctly train-only and correctly sourced from
`data/interim/02_integrated` -- it is the one place the code does what section 2.1's
caveat says the rest of the pipeline should also have done.)

### 2.6 Final feature list per model

- **LR (feature set F3)** = F0 (37) + range_p95_p5 + d_loudness_mean_abs +
  d_bandwidth_mean_abs + d_mfcc1_mean_abs = **41 features**.
- **RF (feature set F2)** = F0 (37) + d_loudness_mean_abs + d_bandwidth_mean_abs +
  d_mfcc1_mean_abs = **40 features** (confirmed by `results/ml/rf_feature_importance.csv`
  having exactly 40 rows).

---

## 3. Cross-validation

Source: `results/ml/cv_results.csv`, `results/ml_metrics.txt` lines ~127-138.

Rule stated before any score is seen: each model (LR, RF) picks its own best feature set
(F0-F3) by grouped-CV weighted-F1; if they disagree, both are carried through separately.

| Model | Feature set | Params | CV weighted-F1 |
|---|---|---|---|
| LR | F0 | regParam=0.001 | 0.5808 |
| LR | F0 | regParam=0.01 | 0.5679 |
| LR | F0 | regParam=0.1 | 0.5121 |
| LR | F1 | regParam=0.001 | 0.5885 |
| LR | F1 | regParam=0.01 | 0.5763 |
| LR | F1 | regParam=0.1 | 0.5192 |
| LR | F2 | regParam=0.001 | 0.5973 |
| LR | F2 | regParam=0.01 | 0.5895 |
| LR | F2 | regParam=0.1 | 0.5390 |
| LR | **F3** | **regParam=0.001** | **0.6003 (best)** |
| LR | F3 | regParam=0.01 | 0.5924 |
| LR | F3 | regParam=0.1 | 0.5432 |
| RF-stage1-fixed (100,10) | F0 | fixed | 0.5599 |
| RF-stage1-fixed (100,10) | F1 | fixed | 0.5584 |
| RF-stage1-fixed (100,10) | **F2** | fixed | **0.5612 (best of the 4, picks F2 for stage 2)** |
| RF-stage1-fixed (100,10) | F3 | fixed | 0.5603 |
| RF (stage 2 grid, on F2) | F2 | numTrees=100,maxDepth=10 | 0.5612 |
| RF (stage 2 grid, on F2) | F2 | numTrees=150,maxDepth=10 | 0.5632 |
| RF (stage 2 grid, on F2) | F2 | numTrees=100,maxDepth=12 | 0.5707 |
| RF (stage 2 grid, on F2) | F2 | **numTrees=150,maxDepth=12** | **0.5746 (best)** |

**Chosen configs:** LR regParam=0.001 on F3; RF numTrees=150, maxDepth=12 on F2.
LR and RF chose **different** feature sets -- both kept and reported separately.

**RF grid trim and why:** originally planned numTrees={100,200} x maxDepth={10,15},
trimmed to numTrees={100,150} x maxDepth={10,12} after the heaviest configuration
(200 trees, depth 15) repeatedly caused `OutOfMemoryError` in the grouped out-of-fold
loop on this machine's ~12GB RAM, local[3], 4g heap (`build.sbt` line 39, `MlConfig.scala`
lines 47-52). Decided before any test-set evaluation -- a hardware constraint, not a
modelling choice.

**Code (`main/ml/ModelTraining.scala`, the leakage-critical CrossValidator call):**
```scala
val grid = new ParamGridBuilder().addGrid(lr.regParam, MlConfig.LrRegParams.toArray).build()
val cv = new CrossValidator()
  .setEstimator(buildPipeline(cols, lr)).setEvaluator(evaluator).setEstimatorParamMaps(grid)
  .setFoldCol(MlConfig.FoldColName).setNumFolds(MlConfig.NumFolds).setSeed(MlConfig.Seed)
val cvModel = cv.fit(trainDf)
```

**Code (`main/ml/ModelTraining.scala`, the pipeline shape):**
```scala
def buildPipeline(featureCols: Seq[String], classifier: PipelineStage): Pipeline = {
  val assembler = new VectorAssembler().setInputCols(featureCols.toArray).setOutputCol("features")
  val scaler = new StandardScaler()
    .setInputCol("features").setOutputCol("scaledFeatures")
    .setWithMean(true).setWithStd(true)
  new Pipeline().setStages(Array(assembler, scaler, classifier))
}
```

---

## 4. Results table (model x set x level)

Source: `results/ml/metrics_summary.csv` (identical to `results/ml_metrics.txt` lines
1614-1660). All values below are copied verbatim.

| Model | Level | Set | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|---|
| LR | window | train | 0.6679 | 0.6645 | 0.6638 |
| LR | recordingSoft | train | 0.7776 | 0.7742 | 0.7738 |
| LR | recordingMajority | train | 0.7631 | 0.7596 | 0.7591 |
| LR | window | oof | 0.6042 | 0.6013 | 0.6006 |
| LR | recordingSoft | oof | 0.6875 | 0.6841 | 0.6834 |
| LR | recordingMajority | oof | 0.6846 | 0.6809 | 0.6804 |
| LR | window | test | 0.6191 | 0.6213 | 0.6201 |
| **LR** | **recordingSoft** | **test** | **0.7027** | **0.7045** | **0.7036** |
| LR | recordingMajority | test | 0.7196 | 0.7214 | 0.7206 |
| RF | window | train | 0.9612 | 0.9613 | 0.9612 |
| RF | recordingSoft | train | 0.9927 | 0.9928 | 0.9927 |
| RF | recordingMajority | train | 0.9913 | 0.9913 | 0.9913 |
| RF | window | oof | 0.5788 | 0.5759 | 0.5751 |
| RF | recordingSoft | oof | 0.6308 | 0.6260 | 0.6255 |
| RF | recordingMajority | oof | 0.6323 | 0.6281 | 0.6275 |
| RF | window | test | 0.6028 | 0.6047 | 0.6035 |
| **RF** | **recordingSoft** | **test** | **0.6824** | **0.6845** | **0.6833** |
| RF | recordingMajority | test | 0.6824 | 0.6844 | 0.6834 |
| Majority | window | train | 0.1018 | 0.0185 | 0.0188 |
| Majority | recordingSoft | train | 0.1017 | 0.0185 | 0.0188 |
| Majority | recordingMajority | train | 0.1017 | 0.0185 | 0.0188 |
| Majority | window | oof | 0.1003 | 0.0308 | 0.0312 |
| Majority | recordingSoft | oof | 0.1003 | 0.0308 | 0.0312 |
| Majority | recordingMajority | oof | 0.1003 | 0.0308 | 0.0312 |
| Majority | window | test | 0.1014 | 0.0184 | 0.0187 |
| Majority | recordingSoft | test | 0.1014 | 0.0184 | 0.0187 |
| Majority | recordingMajority | test | 0.1014 | 0.0184 | 0.0187 |
| CentroidPlain | window | train | 0.5462 | 0.5273 | 0.5261 |
| CentroidPlain | recordingSoft | train | 0.5465 | 0.5276 | 0.5265 |
| CentroidPlain | recordingMajority | train | 0.5465 | 0.5276 | 0.5265 |
| CentroidPlain | window | oof | 0.5393 | 0.5207 | 0.5194 |
| CentroidPlain | recordingSoft | oof | 0.5392 | 0.5205 | 0.5194 |
| CentroidPlain | recordingMajority | oof | 0.5392 | 0.5205 | 0.5194 |
| CentroidPlain | window | test | 0.5237 | 0.4957 | 0.4939 |
| CentroidPlain | recordingSoft | test | 0.5236 | 0.4956 | 0.4939 |
| CentroidPlain | recordingMajority | test | 0.5236 | 0.4956 | 0.4939 |
| CentroidPCA | window | train | 0.6989 | 0.6946 | 0.6938 |
| CentroidPCA | recordingSoft | train | 0.6991 | 0.6948 | 0.6941 |
| CentroidPCA | recordingMajority | train | 0.6991 | 0.6948 | 0.6941 |
| CentroidPCA | window | oof | 0.6470 | 0.6429 | 0.6419 |
| CentroidPCA | recordingSoft | oof | 0.6468 | 0.6426 | 0.6416 |
| CentroidPCA | recordingMajority | oof | 0.6468 | 0.6426 | 0.6416 |
| CentroidPCA | window | test | 0.6453 | 0.6423 | 0.6409 |
| CentroidPCA | recordingSoft | test | 0.6453 | 0.6422 | 0.6408 |
| CentroidPCA | recordingMajority | test | 0.6453 | 0.6422 | 0.6408 |

RF train (recordingSoft) 0.9927 vs. OOF 0.6308 vs. test 0.6824 is the headline
overfitting illustration -- train accuracy alone would badly overstate RF.

---

## 5. Test set, recording level: per-class and confusion matrices (LR, RF)

Source: `results/ml/per_class_LR_recordingSoft_test.csv`, `confusion_LR_recordingSoft_test.csv`,
`per_class_RF_recordingSoft_test.csv`, `confusion_RF_recordingSoft_test.csv`.

### LR, per-class (test, recording-level, soft voting)

| Genre | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| blues | 0.5714 | 0.8000 | 0.6667 | 30 |
| classical | 1.0000 | 0.9000 | 0.9474 | 30 |
| country | 0.6129 | 0.6333 | 0.6230 | 30 |
| disco | 0.6176 | 0.7000 | 0.6563 | 30 |
| hiphop | 0.6250 | 0.5172 | 0.5660 | 29 |
| jazz | 0.6286 | 0.7333 | 0.6769 | 30 |
| metal | 0.9615 | 0.8929 | 0.9259 | 28 |
| pop | 0.7308 | 0.6552 | 0.6909 | 29 |
| reggae | 0.7778 | 0.7000 | 0.7368 | 30 |
| rock | 0.6250 | 0.5000 | 0.5556 | 30 |

### LR, confusion matrix (test, recording-level, rows = true, cols = predicted)

| True\Pred | blues | classical | country | disco | hiphop | jazz | metal | pop | reggae | rock |
|---|---|---|---|---|---|---|---|---|---|---|
| blues | 24 | 0 | 1 | 0 | 0 | 4 | 1 | 0 | 0 | 0 |
| classical | 0 | 27 | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| country | 4 | 0 | 19 | 3 | 0 | 2 | 0 | 0 | 0 | 2 |
| disco | 0 | 0 | 0 | 21 | 2 | 1 | 0 | 0 | 2 | 4 |
| hiphop | 3 | 0 | 0 | 3 | 15 | 1 | 0 | 5 | 2 | 0 |
| jazz | 6 | 0 | 1 | 0 | 0 | 22 | 0 | 0 | 0 | 1 |
| metal | 1 | 0 | 0 | 1 | 0 | 0 | 25 | 0 | 0 | 1 |
| pop | 0 | 0 | 4 | 0 | 3 | 2 | 0 | 19 | 1 | 0 |
| reggae | 0 | 0 | 1 | 2 | 4 | 1 | 0 | 0 | 21 | 1 |
| rock | 4 | 0 | 3 | 4 | 0 | 1 | 0 | 2 | 1 | 15 |

Row sums: 30,30,30,30,29,30,28,29,30,30 = 296 -- matches the 296 test recordings exactly.

### RF, per-class (test, recording-level, soft voting)

| Genre | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| blues | 0.6757 | 0.8333 | 0.7463 | 30 |
| classical | 1.0000 | 0.9000 | 0.9474 | 30 |
| country | 0.4286 | 0.6000 | 0.5000 | 30 |
| disco | 0.5556 | 0.6667 | 0.6061 | 30 |
| hiphop | 0.8500 | 0.5862 | 0.6939 | 29 |
| jazz | 0.6129 | 0.6333 | 0.6230 | 30 |
| metal | 0.8519 | 0.8214 | 0.8364 | 28 |
| pop | 0.7000 | 0.7241 | 0.7119 | 29 |
| reggae | 0.8148 | 0.7333 | 0.7719 | 30 |
| rock | 0.5263 | 0.3333 | 0.4082 | 30 |

### RF, confusion matrix (test, recording-level)

| True\Pred | blues | classical | country | disco | hiphop | jazz | metal | pop | reggae | rock |
|---|---|---|---|---|---|---|---|---|---|---|
| blues | 25 | 0 | 2 | 0 | 0 | 0 | 1 | 0 | 1 | 1 |
| classical | 0 | 27 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| country | 5 | 0 | 18 | 4 | 0 | 3 | 0 | 0 | 0 | 0 |
| disco | 0 | 0 | 1 | 20 | 1 | 1 | 2 | 2 | 1 | 2 |
| hiphop | 1 | 0 | 2 | 4 | 17 | 0 | 1 | 2 | 1 | 1 |
| jazz | 5 | 0 | 3 | 1 | 0 | 19 | 0 | 2 | 0 | 0 |
| metal | 0 | 0 | 0 | 0 | 0 | 1 | 23 | 0 | 0 | 4 |
| pop | 0 | 0 | 2 | 0 | 2 | 2 | 0 | 21 | 1 | 1 |
| reggae | 0 | 0 | 4 | 2 | 0 | 1 | 0 | 1 | 22 | 0 |
| rock | 1 | 0 | 10 | 5 | 0 | 1 | 0 | 2 | 1 | 10 |

Row sums: 30,30,30,30,29,30,28,29,30,30 = 296 -- matches. Rock is RF's weakest genre
(recall 0.33, 10/30 correct, confused mostly with country).

---

## 6. Genre-pair check and centroid column sums

Source: `results/ml/genre_pair_check.csv`, `results/ml/centroid_column_sums.csv`.

| Check (test set, recording level) | LR | RF |
|---|---|---|
| hiphop -> reggae | 2 | 1 |
| reggae -> hiphop | 4 | 0 |
| country -> rock | 2 | 0 |
| rock -> country | 3 | 10 |
| rock total off-diagonal | **15** (highest, least separable) | **20** |
| classical total off-diagonal | **3** (lowest, most distinct) | **3** |
| country -> blues | 4 | 5 |
| disco -> pop | 0 | 2 |
| hiphop -> pop | 5 | 2 |
| rock -> metal | 0 | 0 |
| rock -> blues | 4 | 1 |

This matches Phase 3's "rock is the most central/confusable genre, classical the most
distinct" claim. For Phase 3's closest-pair claims (hiphop<->reggae, country<->rock),
LR shows confusion in both directions for both pairs; RF only one direction (reggae->hiphop
= 0, country->rock = 0 for RF).

**Centroid column sums (predicted-count per genre, test set):**

| Genre | Plain (unwhitened) | PCA-whitened |
|---|---|---|
| blues | **75** | **55** |
| classical | 30 | 28 |
| country | 19 | 27 |
| disco | 11 | 30 |
| hiphop | 15 | 23 |
| jazz | 17 | 21 |
| metal | 36 | 28 |
| pop | **49** | **32** |
| reggae | 28 | 28 |
| rock | 16 | 24 |

Confirms the "pop/blues attractor" hypothesis: plain centroid over-predicts blues (75/296
= 25%) and pop (49/296 = 17%); PCA-whitening brings both down toward the ~29.6-per-genre
baseline (55 and 32).

---

## 7. Feature importances and coefficients

Source: `results/ml/rf_feature_importance.csv` (40 rows), `results/ml/lr_coefficients.csv`.

### Top 15 RF feature importances

| Rank | Feature | Importance |
|---|---|---|
| 1 | d_loudness_mean_abs | 0.091115 |
| 2 | rms_mean | 0.076794 |
| 3 | d_mfcc1_mean_abs | 0.066478 |
| 4 | d_bandwidth_mean_abs | 0.064298 |
| 5 | log_rms_var | 0.054166 |
| 6 | mfcc4_mean | 0.050256 |
| 7 | spectral_bandwidth_mean | 0.040174 |
| 8 | mfcc1_mean | 0.039517 |
| 9 | log_flatness_mean | 0.037033 |
| 10 | log_flatness_var | 0.035887 |
| 11 | rolloff_mean | 0.027142 |
| 12 | mfcc9_mean | 0.023082 |
| 13 | mfcc3_mean | 0.022963 |
| 14 | mfcc6_mean | 0.021844 |
| 15 | mfcc8_mean | 0.021671 |

3 of the top 4 are the change features (d_loudness_mean_abs, d_mfcc1_mean_abs,
d_bandwidth_mean_abs); rms_mean (rank 2) is a base feature, not engineered. This still
supports the "extra features earned their place under grouped CV" claim.

### LR top +/- standardised coefficients, 4 notable genres

| Genre | Top positive | Top negative |
|---|---|---|
| blues | mfcc3_mean (+0.839), rms_mean (+0.779), zcr_mean (+0.719) | log_flatness_var (-0.871), range_p95_p5 (-0.840), mfcc7_mean (-0.764) |
| classical | range_p95_p5 (+1.268), log_zcr_var (+0.705), mfcc11_mean (+0.669) | mfcc4_mean (-1.086), log_spectral_centroid_var (-1.041), log_flatness_var (-1.027) |
| metal | mfcc4_mean (+1.843), log_flatness_mean (+1.119), log_flatness_var (+1.058) | (not in top-3 negative pull) |
| rock | mfcc2_mean (+0.861), log_flatness_mean (+0.764), mfcc1_mean (+0.572) | d_loudness_mean_abs (-0.482), log_spectral_bandwidth_var (-0.465), log_rms_var (-0.426) |

Classical's strongest positive driver is `range_p95_p5` (the recording-level dynamic
range extra feature) -- consistent with Phase 3/4's dynamic-range finding for classical.

---

## 8. Runtime, Spark settings, machine limits

Source: `results/ml_metrics.txt` lines 5, 133, 137, 1995; `build.sbt`; `main/ml/MlConfig.scala`.

| Setting | Value |
|---|---|
| Spark master | local[3] |
| shuffle.partitions | 8 |
| Heap (`run / javaOptions`) | -Xmx4g |
| Machine RAM (stated limit) | ~12 GB |
| Seed | 42 |

| Stage | Wall time |
|---|---|
| Stage 1 (LR 3-config x RF 1 fixed config, x4 feature sets, 5-fold grouped CV) | 1613.5 s (~27 min) |
| Stage 2 (RF numTrees x maxDepth grid on F2, 5-fold grouped CV) | 2013.0 s (~34 min) |
| **Total run** | **5174.8 s (~86.2 min)** |

The out-of-fold loop (25 extra fits: LR, RF, Majority, CentroidPlain, CentroidPCA x 5
folds) and the final-fit stage are not separately timed in the report -- only stage 1,
stage 2, and the grand total are logged. (Listed under "could not verify" below.)

---

## 9. Code snippets (exact, one per stage)

**Split (`main/ml/Split.scala` lines 51-57):** see section 1 above.

**Grouped folds (`main/ml/Split.scala` lines 82-86):** see section 1 above.

**Pipeline (`main/ml/ModelTraining.scala` lines 38-44):** see section 3 above.

**CrossValidator with foldCol (`main/ml/ModelTraining.scala` lines 74-78):** see section 3 above.

**Extra features (`main/ml/ExtraFeatures.scala` lines 43-54):**
```scala
def robustDynamicRange(windows: DataFrame): DataFrame = {
  val withLoudness = windows.withColumn(
    "loudness_db",
    lit(20.0) * log10(col("rms_mean") + AudioConfig.Epsilon)
  )
  withLoudness.groupBy("track_id").agg(
    (percentile_approx(col("loudness_db"), lit(0.95), lit(10000))
      - percentile_approx(col("loudness_db"), lit(0.05), lit(10000))
    ).as(RangeCol)
  )
}
```

**Soft voting (`main/ml/MlMetrics.scala` lines 28-34):**
```scala
def softVoteToRecording(windowPreds: DataFrame): DataFrame = {
  val recordingProbs = windowPreds.groupBy("track_id")
    .agg(first("label_idx").as("label_idx"), Summarizer.mean(col("probability")).as("probability"))
  val argmaxUdf = udf((v: Vector) => v.toArray.zipWithIndex.maxBy(_._1)._2.toDouble)
  recordingProbs.withColumn("prediction", argmaxUdf(col("probability")))
}
```

**Metric computation (`main/ml/MlMetrics.scala` lines 63-91, confusion-matrix-from-counts
core):**
```scala
val confusion = Array.fill(n, n)(0L)
predictions.groupBy(labelCol, predCol).count().collect().foreach { r =>
  val trueIdx = r.getAs[Double](labelCol).toInt
  val predIdx = r.getAs[Double](predCol).toInt
  confusion(trueIdx)(predIdx) = r.getAs[Long]("count")
}
```

**Grouped out-of-fold loop (`main/ml/ModelTraining.scala` lines 160-167):**
```scala
def runGroupedOof[M](trainDf: DataFrame, foldCol: String, numFolds: Int,
                      fitFn: DataFrame => M, predictFn: (M, DataFrame) => DataFrame): DataFrame = {
  val perFoldPredictions = (0 until numFolds).map { fold =>
    val foldTrain   = trainDf.filter(col(foldCol) =!= fold)
    val foldHoldout = trainDf.filter(col(foldCol) === fold)
    val model = fitFn(foldTrain)
    predictFn(model, foldHoldout)
  }
  perFoldPredictions.reduce(_ unionByName _)
}
```

---

## 10. Limitations and deviations from the original plan

1. **RF grid trimmed** from {100,200}x{10,15} to {100,150}x{10,12} after repeated
   `OutOfMemoryError` crashes in the grouped OOF loop (~12GB RAM, local[3], 4g heap).
   Decided before any test-set evaluation -- a hardware limit, not a modelling choice.
2. **Feature-set provenance deviation (new finding, see section 2.1):** `CLAUDE.md`'s
   Phase 5 rules explicitly direct starting feature selection from
   `data/interim/02_integrated` and recomputing eta²/correlation **and skewness**
   train-only. The implemented pipeline instead reads `data/processed/final` (Phase 2's
   already-reduced, already-log-transformed, full-dataset-derived output) for every
   modelling step, and only uses `data/interim/02_integrated` for the narrow
   spectral_centroid_mean recheck. Eta²/correlation ARE correctly recomputed on
   training rows on top of that table, but the log-transform choice and the original
   37-vs-38 feature drop were never redone train-only. This is a genuine, if probably
   small, leakage path and a documented deviation from the written plan.
3. **Determinism claim is undocumented beyond a note.** `notes/phase5_report_material.md`
   states two full runs produced byte-identical outputs, but no diff, checksum, or
   second run's artifacts are retained under `results/` to independently verify this.
4. Window-level `loudness_variation` does not reflect genre as strongly as recording-level
   stats (Phase 3/4 finding, not a Phase 5 issue, but relevant context for interpreting
   `loudness_variation`'s modest eta²=0.1525).
5. RF's search is less thorough than LR's (1 fixed config in stage 1 vs LR's 3-point grid,
   due to RF's cost) -- stated as an explicit limitation in the code's own comments.

---

## 11. Could not verify directly in this audit

- **Determinism (two identical runs):** claimed in `notes/phase5_report_material.md`,
  no artifact (diff/hash/second output set) exists under `results/` to check against.
- **Per-sub-stage runtime** for the out-of-fold loop (25 fits) and the final-fit stage:
  not separately logged; only stage 1, stage 2, and the grand total are in
  `results/ml_metrics.txt`.
- **AUTHORS.md** is UTF-16-encoded and did not render as plain text when read; its content
  (team roster) was not verifiable in this pass beyond what's already duplicated in the
  README's Team section.
- **Whether `docs/` is supposed to contain the written report**: the folder exists but is
  empty except for a 0-byte `.txt` file; no report document was found there or elsewhere
  in the repo.
