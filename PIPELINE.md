# Execution interfaces — Measurement-aware retrospective ICU prediction

This guide describes the preserved source interfaces by reading the code. No research module was imported, no command was run, and no performance or portability validation is implied. Use the original project instructions for the scientific run order.

## Data → analysis → outputs

**Input contract:** Credentialed MIMIC-IV 3.1 and eICU 2.0. Set ICU_RAW_ROOT and ICU_EXTRACT_ROOT locally; generated rows and models remain private.

1. Acquire inputs from their original providers and configure the paths defined in the source.
2. Install the recorded dependencies in an isolated environment.
3. Preserve grouping, seeds, development/evaluation separation and any protocol/hash gates.
4. Run acquisition/extraction and preprocessing before model development. Run evaluation only after the selection/freeze required by the original implementation.
5. Keep raw data, row-level derivatives, predictions and fitted models outside the public repository.

## Project entry sequence

```powershell
python extract.py
python prepare.py
python models.py
python tuning_extension/run_search.py
python tuning_extension/evaluate_locked.py
python integrate_final_analysis.py
```

These commands are documented from the saved project interfaces; they were not executed for this release. Source-specific preconditions still apply.

## Script reference

### `audit.py`

[Open source](audit.py) · Executable entry point

### `coverage_hospital.py`

[Open source](coverage_hospital.py) · Executable entry point

### `evaluate.py`

[Open source](evaluate.py) · Executable entry point

### `extract.py`

[Open source](extract.py) · Supporting module

Raw-data harmonisation. Local restricted outputs are never included in submission ZIP.

### `figures.py`

[Open source](figures.py) · Executable entry point

### `integrate_final_analysis.py`

[Open source](integrate_final_analysis.py) · Supporting module

Recompute dependent diagnostics for the locked final models; no selection.

### `models.py`

[Open source](models.py) · Executable entry point

### `negative_controls.py`

[Open source](negative_controls.py) · Executable entry point

Repeat all shuffled-label controls, retaining every permutation.

### `prepare.py`

[Open source](prepare.py) · Executable entry point

### `recency_audit.py`

[Open source](recency_audit.py) · Executable entry point

Outcome-independent sample audit of raw predictor timing; no label-based sampling.

### `sensitivity.py`

[Open source](sensitivity.py) · Executable entry point

### `tuning_extension/evaluate_locked.py`

[Open source](tuning_extension/evaluate_locked.py) · Executable entry point

### `tuning_extension/run_search.py`

[Open source](tuning_extension/run_search.py) · Executable entry point

### `tuning_extension/verify_extension.py`

[Open source](tuning_extension/verify_extension.py) · Executable entry point

## Reproduction limits

Absolute paths, frozen-file references and platform-specific assumptions are preserved. Configure a matching local workspace rather than bypassing scientific integrity guards. A repository containing existing scripts is not evidence that every data or environment prerequisite is satisfied. No original clinical results or completed-run artifacts are redistributed.
