# Tool and source disclosure

## AI tools used

- Claude (Anthropic), model Claude Opus 5.5, in the claude.ai chat with its code-execution sandbox.

## What it assisted with

**Summary:** all code and tests were drafted by Claude to speed up the process. I validated them (running the tests and the demo, reviewing the code), and the changes I asked for were implemented by Claude. The main ones: easy-to-follow code with explanations at every step, a simpler training step (later also without bootstrap intervals, a separate metrics file or field-by-field row checks), the engine reduced to a prototype at the timebox, and finally fitting, preprocessing and metrics switched to scikit-learn, with the model saved as a pickled pipeline plus JSON metadata.

| Area | Assistance |
|---|---|
| Understanding | Explained the brief; checked that the .docx matches README.md and that the data matches the generator's output; profiled the sample data |
| Step 1: input validation and timestamps | Drafted `errors.py`, `_time.py`, `_records.py`, `_io.py` and tests `test_time.py`, `test_records.py`, `test_io.py`, `tests/_factories.py` |
| Step 2: event store | Drafted `_store.py`, `ConflictError`, the `make_event` test factory and `tests/test_store.py`; proposed the exact rule for `ingest`'s True/False and the oracle test that checks it |
| Step 3: features | Drafted `_canonical.py`, `_features.py`, `tests/test_canonical.py` and `tests/test_features.py`; ran the event_at-versus-receipt leak comparison on the sample |
| Step 4: training rows | Drafted `_labels.py`, `_dataset.py`, `read_manifest` in `_io.py`, `build_training_rows` in `solution.py`, `tests/test_labels.py` and `tests/test_dataset.py`; ran six generator variants in scratch folders |
| Step 5: training | Drafted `_metrics.py`, `_model.py`, `_training.py`, `ArtifactError`, `write_atomic`/`parse_json_object` in `_io.py`, `train` in `solution.py`, and tests `test_metrics.py`, `test_model.py`, `test_training.py`; ran the shuffled-label and weak-signal checks; later drafted `tools/shuffled_label_check.py` to make the shuffled-label check reproducible |
| Engine prototype | Drafted the prototype `RiskEngine` in `solution.py`, `EventStore.forget`, `RetentionError`/`SnapshotError`, `synthetic_training_rows` and `tests/test_engine.py`. An earlier, fuller engine draft was **not used**, because I could not review it within the timebox |
| Demo and documents | Drafted `tools/demo.py`, CUSTOMER_NOTE.md and the final DECISIONS/WORKLOG/RESULTS updates; ran the demo |

## Representative prompts (up to three)

1. **Step 1 (paraphrased):** "Start the first coding step with full, simple code covering everything we discussed, plus how to run and test it, where each file goes, why each choice was made and what alternatives were rejected, and update the relevant .md files."
   **How I checked it:** ran the full suite with both README commands; read every function against the DECISIONS.md rules; confirmed that breaking a rule makes a test fail (see below).
2. **Engine (paraphrased):** "My 7-hour limit is reaching. For the engine I only want a working prototype that still shows the four customer notes are wrong and that I have a solution; give me your view before building it."
   **How I checked it:** ran the tests and the demo; confirmed each customer-note test fails when its protection is removed (see below); read the engine method by method.
3. **Stress cases (paraphrased):** "Is my code exposed to these hidden-evaluation variants (other seeds, conflicts, offsets, late labels…) and any other breakouts?"
   **How I checked it:** six generator variants were generated into scratch folders (never `data/`) and run through `build_training_rows`; results are in RESULTS.md.

## How I validated the work

- On my machine (Python 3.12.8): ran `python tools/demo.py` with the pure-Python fit. It reproduced `model_version` `logreg-bec4ef1481f681cf`, identical to the sandbox run on Python 3.12.3. The final pickled-pipeline version gives `logreg-7170fa340d6fa4bd`, confirmed on my machine (Python 3.12.8) with the pinned dependencies, identical to the sandbox.
- scikit-learn switch: a test checks that the fit solves log-loss + ½‖w‖² (gradient ≈ 0). The weights match the earlier Newton fit within 0.0000003, and every metric is identical to 6 decimals.
- Pickle artifact: tests confirm that a tampered `model.pkl` is refused before unpickling, that a different recorded scikit-learn version or feature schema is refused, that `model_version` survives a load/save round trip (it hashes parameters, not bytes), and that a fresh process loads the same model.
- Mutation check (assistant's sandbox, Python 3.12.3): five validation rules were reverted one at a time, and each made the suite fail.
- Step 2: a brute-force oracle confirms `changed` for every add (120 delivery orders of a tricky set, plus every sample event). Mutation check: six store rules were broken one at a time, and each made the suite fail.
- Step 3: for every sample decision, features from the full store equal those from a store fed only records received by `as_of`. Mutation check: six feature rules were broken one at a time, and each made the suite fail.
- Step 4: labels and eligibility for every sample row match an independent brute-force check. Six generator variants were run without touching `data/`. Mutation check: seven label/dataset rules were broken one at a time, and each made the suite fail.
- Step 5: average precision matches the textbook example (0.8333). The shuffled-label check (`tools/shuffled_label_check.py`) confirms the evaluation is sound. Mutation check: eight rules; one gap (naive sigmoid) was found and closed with a new test.
- Engine: online/offline parity on all 720 sample decisions. Mutation check: six key protections broken one at a time (notes 1, 3 and 4, plus a torn reload), and each was caught. `python tools/demo.py` ran end to end (output in RESULTS.md).
- My own validation: I ran the code and reviewed it step by step, using Claude's explanation of each component and its rejected alternatives, and asked for changes where I wanted a simpler design.

## Copied code and licenses

- No third-party code was copied.
- Declared dependencies (`pyproject.toml`, pinned): scikit-learn 1.8.0, numpy 2.4.4 and scipy 1.17.1, all BSD-3-Clause licensed. They provide the model pipeline (imputer, scaler, one-hot encoder, logistic regression), its pickled artifact, and the evaluation metrics.
- `contracts.py`, `generate_dataset.py` and `test_public_contract.py` are the supplied starter files.
