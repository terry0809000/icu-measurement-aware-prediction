# Measurement-aware retrospective ICU prediction

**Prediction Modelling · Research pipeline · Source-code release**

This project provides the author's existing analysis implementation for measurement-aware retrospective icu prediction. Scientific source files are preserved from the local research product. This upload organizes code and documentation; it does not represent new experiments, model retraining or an independently verified reproduction.

## Pipeline

1. **Acquire inputs.** Credentialed MIMIC-IV 3.1 and eICU 2.0. Set ICU_RAW_ROOT and ICU_EXTRACT_ROOT locally; generated rows and models remain private.
2. **Prepare the environment.** Use the project-specific dependency files listed below in an isolated environment. Preserve the versions and seeds recorded in the source release.
3. **Prepare data and run the model or analysis.** Follow the original execution notes and entry points below, preserving time order, participant grouping and fixed evaluation boundaries.
4. **Evaluate and summarize.** Use the included evaluation/analysis scripts. Write generated data, predictions, fitted objects and caches to a private working copy.

This is a source-only repository. Raw data, individual-level derivatives and fitted checkpoints are not bundled. Some saved-evidence reproduction commands in the original notes require artifacts from the original source release; obtain those independently where authorized.

## Environment

- [`environment.json`](environment.json)
- [`requirements.txt`](requirements.txt)

## Execution

See the [pipeline and command reference](PIPELINE.md) for entry points, required arguments and the source-specific input contract.

Run commands from the repository root unless the original notes specify another working directory. Use a fresh copy and configure local data paths before running.

```bash
python extract.py
python prepare.py
python models.py
python tuning_extension/run_search.py
python tuning_extension/evaluate_locked.py
python integrate_final_analysis.py
```

### Original execution notes

- [README.md](docs/original/README.md)

### Scientific source inventory

<details>
<summary>Browse the preserved scientific implementation</summary>

- [`audit.py`](audit.py)
- [`coverage_hospital.py`](coverage_hospital.py)
- [`evaluate.py`](evaluate.py)
- [`extract.py`](extract.py)
- [`figures.py`](figures.py)
- [`integrate_final_analysis.py`](integrate_final_analysis.py)
- [`models.py`](models.py)
- [`negative_controls.py`](negative_controls.py)
- [`prepare.py`](prepare.py)
- [`recency_audit.py`](recency_audit.py)
- [`sensitivity.py`](sensitivity.py)
- [`statistics_core.py`](statistics_core.py)
- [`tuning_extension/evaluate_locked.py`](tuning_extension/evaluate_locked.py)
- [`tuning_extension/run_search.py`](tuning_extension/run_search.py)
- [`tuning_extension/verify_extension.py`](tuning_extension/verify_extension.py)

</details>

## Repository contents and rights

Only scientific code, dependencies, configuration and execution documentation are included. Manuscripts, submission documents, raw participant records, trained models, credentials, caches and software environments are excluded. Notebook cell outputs and execution counters are cleared for publication; code cells are retained.

Existing source and third-party license notices govern their respective files. No blanket license is invented for material whose original release did not specify one. Original dataset licenses and access requirements continue to apply.

Research software is not a clinical deployment claim. Negative findings and the source release's interpretation limits remain applicable. No tests or experiments were run during this upload, as requested by the author.
