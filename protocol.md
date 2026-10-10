# Measurement and validation audit: revised protocol

Recorded 19 September 2026, before fitting this revision. This is a post hoc redesign after earlier inspection of both databases, not preregistration or a previously untouched validation. No constraint is imposed on the ordering of internal and external AUCs.

## Primary question and estimand
Among adults alive and still in an index ICU stay at 24 hours, how do four fitted model families compare for death before discharge from that index ICU stay, with the same available inputs, training sample size, tuning opportunities and evaluation patients? Live ICU discharge is the non-event for this precisely defined in-unit endpoint. The follow-up lasts until ICU exit; there is no fixed future prediction horizon. This is not hospital mortality or mortality after ICU discharge. It excludes early deaths by design and does not estimate an admission-time risk.

The previous vasopressor composite is retired from primary inference because infusion interfaces do not establish equivalent negative labels. Outcome scores from this changed task are not improvements over the previous task.

## Cohort and partition
Use the existing audited adult raw extraction, but no vasopressor exclusion. Require stay length >24 hours and no recorded ICU death <=24. Exclude eICU unknown discharge status. Choose exactly one eligible stay per patient by a deterministic outcome-independent SHA256 rank; do not call this the first chronological admission. Assign eICU hospitals (not rows) to train/tune/probability-calibration/conformal/test in 50/10/10/10/20 nominal proportions; remove selected patients whose other eligible stays cross those hospital roles. Assign MIMIC patients to the same proportions by hash. Report realized counts and exclusions.

## Inputs and comparison
Six concepts: heart rate, respiratory rate, SpO2, noninvasive SBP/DBP/MAP. Six hourly bins ending at 18,19,20,21,22,23 hours. No input from (23,24]; the outcome is subsequent recorded ICU death before discharge. Keep last available bounded observation per bin, no forward/backward fill, train-only medians and scales. Include missing masks, age, male and unknown-sex indicators. Both model types receive exactly the same values and masks. Raw measurement counts and within-hour last-reading recency are audited; common concepts do not imply identical measurement mechanisms.

Train in BOTH directions independently. Use the smaller of the two training patient counts, selected by an outcome-independent hash, for both source domains and every model. Source-only tuning, probability and conformal calibration. Never tune or select the source model using target test labels. Source test and target test are the same fixed respective populations across all systems. Source test on eICU is a held-out-hospital evaluation, not a random same-hospital internal test.

Four families: logistic regression, histogram gradient boosting, extra trees, compact Transformer. Eight disclosed configurations each. No architecture-specific row downsampling. Transformer uses 30 fixed epochs, final checkpoint, no test-selected epoch. Select by source-tune AP then AUROC; primary seed42, repeat chosen configuration seeds117 and2026. Equal rows and candidate counts are not equal compute or proof of optimal architecture performance.

## Inference
AUROC, AP with prevalence baseline, Brier, log loss, observed/expected and calibration slope/intercept. Calibration-in-the-large uses offset slope fixed1 and is distinguished from the free-slope intercept. Paired 1,000 patient bootstrap across models, hospital bootstrap sensitivity on eICU. Independent source/target bootstrap estimates external-minus-source AUC; no paired cross-database test. Cluster intervals condition on fitted models. Holm adjustment by direction/population/metric. Report all candidate, seed and subgroup results.

## Diagnostic analyses, not model selection
1. Covariate-overlap weighted evaluation using a domain classifier fitted to equally sampled unlabelled development inputs; report domain AUC, overlap and effective sample size. Weights are clipped domain probabilities [0.05,0.95], source weights p(domain=eICU), target weights 1-p. These define an overlap population, not a causal correction or guarantee of equal case mix.
2. Symmetric outcome-blind thinning: for each feature/hour use the lower observation frequency in the two development samples. Randomly remove observed test values to that rate, three fixed seeds, frozen models and source preprocessing; report achieved availability and resulting metrics. This is an artificial stress test, not a deployable harmonization or a causal experiment.
3. Same-population evaluation restricted to at least half the six hourly bins present for each core vital (HR/RR/SpO2), and modality ablation excluding BP; show changed denominators.
4. HGB refits without explicit masks, without BP, and alternate short-horizon death labels and discharge-offset-aligned death labels, equal candidate budgets and source-only tuning.
5. Age/sex-only baseline and shuffled-training-label negative control. Explicit ID-disjointness, exact time-boundary, calibration separation and sklearn metric checks.
6. Separate probability updating using target calibration patients (hospitals distinct from eICU test), same frozen target test. Marginal and class-conditional conformal, event-class coverage and set size, Wilson intervals; no cross-domain coverage guarantee.

## Reporting
Replace the manuscript's numerical results with this study. Report actual results including failure; do not choose endpoints, masks, seeds or subgroups to force internal AUC above external AUC. Explain internal-versus-external distinctions and residual ascertainment. Preserve >=40 relevant verified references and a 20-dimension literature-derived question/answer audit. No independent review, prospective validation, ethics exemption, publication guarantee or unperformed experiment is claimed.

## Timestamp-driven amendment before test performance inspection

An audit after beginning candidate fits, but before reviewing their test metrics, found MIMIC death-to-ICU-exit delay median2.77h (6519 recorded ICU deaths; 169 deaths in (24,48] with exit after48h). eICU death_hour is ICU discharge offset. The primary task is therefore death before index ICU discharge, conditional on being recorded in ICU and not known dead at24h. Interrupted short-horizon models are isolated and not reused. MIMIC can exclude documented deaths before24h more accurately; eICU cannot resolve unrecorded death-to-exit delay. This remains a stated limitation. The difference is audited and secondary endpoints include the native and exit-offset 24h labels.

## Negative-control diagnostic expansion

After the first shuffled-label control yielded a below-chance result in one held-out population, the control was expanded to20 fixed label permutations per source (seeds1942 through1961). Every permutation is retained; no permutation selects a model or changes a primary estimate. This assesses variability of the diagnostic rather than presenting one random fit as definitive.

## Negative-control diagnostic expansion

After the first shuffled-label control yielded a below-chance result in one held-out population, the control was expanded to20 fixed label permutations per source (seeds1942 through1961). Every permutation is retained; no permutation selects a model or changes a primary estimate. This assesses variability of the diagnostic rather than presenting one random fit as definitive.
