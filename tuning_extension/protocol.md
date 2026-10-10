# AUROC focused tuning extension

This amendment is recorded before executing the expanded search on 19 September 2026. It follows examination of v10 results and a request to expand hyperparameter tuning. It is a post hoc robustness analysis, not a prospectively preregistered study or a fresh external validation. Original v10 analyses and outputs remain unchanged.

The endpoint, feature extraction, patient selection, original partitions and 27,437 patient final training budget per source remain fixed. No endpoint, cohort, missingness rule or test partition may be changed to improve performance. The original primary analysis selected on average precision; this extension selects on AUROC to answer a separate optimisation question.

## Selection before test evaluation

For each source and each of four original model families, screen 16 predetermined configurations. Eight reproduce the original configuration grid and eight explore additional regularisation, learning rate, capacity or training duration. The full grid is written to search_space.json before fitting.

Screening uses a new outcome-independent approximately 20% split within the original matched training sample. MIMIC splits by patient and eICU by hospital using SHA256 and a fixed salt. Preprocessing is fitted on the remaining inner training patients only. All families use identical source-specific inner training and screening patients. Stage-one training sizes can differ between sources due to hospital grouping; final full-budget training is matched. No original tuning, calibration, conformal or test patients enter screening.

Screen every configuration with seed42. Rank on screening AUROC, then average precision, then ascending configuration identifier. Promote the best two configurations per family. Refit each on all 27,437 original matched training patients with seeds42,117,2026. Fit preprocessing on that full training set. Rank finalists by mean original source-tuning AUROC across three seeds, then mean AP, then configuration identifier. Select the family by the same mean validation metrics. Keep seed42 as the primary fit; report both additional seeds without choosing a best seed or constructing an unplanned ensemble. Report raw training performance and training-validation gaps as overfitting diagnostics, not as optimisation targets.

This two-stage search reduces direct reuse of the confirmation validation set across all configurations, but cannot eliminate selection optimism or prior investigator knowledge of these resources. Three seeds measure stochastic fitting variation, not validation sampling uncertainty. No claim of the globally optimal hyperparameters is permitted. Equal candidate counts are not equal computational effort.

Lock all family configurations and source-selected families in a timestamped, hashed manifest before computing any source or target test predictions for the extension. Calibration and conformal fitting occur only after selection. Source calibration uses the original source probability-calibration patients; source conformal thresholds use its distinct conformal partition. Target updating, if reported, uses separate target calibration and conformal partitions and is labelled adaptation.

## Evaluation

Evaluate every locked family and seed on the original source and target tests. Report AUROC, AP, Brier score, log loss, calibration, marginal and class-conditional set coverage. Bootstrap seed42 AUROC/AP/Brier estimates with 1,000 common patient resamples and a separate eICU hospital bootstrap. Pair extension-versus-v10 comparisons on identical test patients. Compare the newly source-selected seed42 system against the originally source-selected seed42 system, and each family against its v10 counterpart. Report differences with intervals; do not choose a final system based on those tests.

The same previously evaluated test populations are retained for comparability and are explicitly described as reused retrospective benchmarks. Neither resplitting previously used records nor hiding old results would restore untouched external validation. No further search is triggered by test results from this extension.

## Scientific design assessment

The primary endpoint is recorded index ICU mortality after a 24-hour landmark, not a fixed-horizon early-warning task or an intervention effect. Live discharge ends follow-up as a non-event. Exact eICU death time remains unavailable; known versus unrecorded death-to-exit delays cause residual eligibility uncertainty. Retain these limitations rather than changing the task to force a performance narrative.

Validate patient uniqueness, hospital separation, training-only fitting of preprocessing, event support in each selection partition, finite predictions and exact preservation of held-out roles. Preserve all attempted configurations, exceptions and seed results. Changes to this protocol after starting must be recorded as amendments, not silently edited.
