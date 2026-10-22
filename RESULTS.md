# Results

Every number here comes from an actual run. The test suite and the end-to-end demo (`python tools/demo.py`) were run on my machine (Python 3.12.8); results marked as development checks were run once in the AI assistant's sandbox.

## Sample data is unmodified

```bash
python tools/generate_dataset.py --seed 1729 --size 240 --output /tmp/regen
for f in events.jsonl labels.jsonl decisions.jsonl MANIFEST.json; do cmp data/$f /tmp/regen/$f && echo "$f identical"; done
```

Result: all four files identical.

## Sample profile

| Item | Count |
|---|---|
| Events (= readings + duplicates + corrections + backdated) | 4,402 (3,840 + 480 + 35 + 47) |
| Labels | 47 |
| Decisions | 720 (714 eligible, 6 censored) |
| Readings received exactly at a decision time | 462 |
| Out-of-order deliveries (next record received earlier) | 175 |
| Conflicts, future device clocks, reversed revisions, post-cutoff records | 0 |

## Test suite

```bash
pytest                                                # or: PYTHONPATH=src python -m unittest discover -s tests
```

Run on my machine (Python 3.12.8):

```
179 passed, 146 subtests passed in 40.37s
```

The assistant's sandbox (Python 3.12.3, `unittest`) also passed all 179 tests.

Before the switch to scikit-learn, my machine (Python 3.12.8) reproduced the pure-Python `model_version` `logreg-bec4ef1481f681cf`. The final pickled-pipeline version gives `logreg-7170fa340d6fa4bd`, identical on my machine (Python 3.12.8) and in the sandbox (Python 3.12.3). (The suite is slower than before because each scikit-learn score takes about 1.5 ms and the parity tests score thousands of times.)

## Event store on the sample (step 2)

| Measure | Result |
|---|---|
| Add outcomes in file order | 3,922 accepted, 480 duplicate, 0 superseded on arrival, 0 conflicts |
| Stored revisions / shipments | 3,922 / 240 |
| Same answers for all 720 decisions across file, reversed and shuffled delivery order | yes |
| Readings used by any decision with `received_at > as_of` | 0 |
| `changed` flag matches the brute-force oracle for all 4,402 adds | yes |

## Features on the sample (step 3)

| Measure | Result |
|---|---|
| Decisions with features | 720 (720 distinct digests) |
| Usable readings in the 6 h window per decision | 3: 38, 4: 205, 5: 382, 6: 95 |
| Decisions with `no_readings_in_window` / without a trend | 0 / 0 |
| Features identical to a store fed only records received by `as_of` | 720 of 720 |
| Decisions whose features include the 19.0 °C backdated reading: receipt-time rule | **0** |
| Same, if availability were (wrongly) based on `event_at` | 94 (and 645 of 720 feature sets would differ) |

## Training rows on the sample (step 4)

Report from `build_dataset` (`observation_end` 2026-02-11T12:00:00Z, horizon 6 h, reporting lag 30 h):

| Input | Counts |
|---|---|
| Events | 4,402 input: 3,922 accepted, 480 duplicate, 0 superseded, 0 conflict, 0 received after cutoff |
| Labels | 47 input: 47 used, 0 duplicate, 0 available after cutoff, 0 beyond reporting lag |
| Decisions | 720 input, 720 rows, 0 merged, **714 eligible, 47 positive**, 6 censored (`label_window_immature`) |

The six censored decisions are those whose label would only be mature after the cutoff:

| Shipment | as_of | Label mature at |
|---|---|---|
| e-00237 | 2026-02-10T02:00:00Z | 2026-02-11T14:00:00Z |
| e-00238 | 2026-02-10T03:00:00Z | 2026-02-11T15:00:00Z |
| e-00239 | 2026-02-10T04:00:00Z | 2026-02-11T16:00:00Z |
| e-00238 | 2026-02-10T06:00:00Z | 2026-02-11T18:00:00Z |
| e-00239 | 2026-02-10T07:00:00Z | 2026-02-11T19:00:00Z |
| e-00239 | 2026-02-10T10:00:00Z | 2026-02-11T22:00:00Z |

## Generator variants (step 4 checks)

Each variant was generated into a scratch folder; `data/` was not touched. These are development checks, run once in the assistant's sandbox; they are not scripted.

| Variant | Result |
|---|---|
| Manifest lag 48 h | stopped: `label_lag_mismatch` |
| Cutoff 30 h earlier | 2 `decision_after_cutoff`, 27 immature; 10 events and 3 labels after the cutoff ignored |
| Labels up to 34 h late | 10 counted as `beyond_reporting_lag` |
| Every decision duplicated | 240 merged, rows unchanged (720) |
| Extra decision at +17 h | 960 rows, 93 positives |
| Incident at +12 h instead of +18 h | 94 positives (now at the +8 h and +11 h decisions) |

## Training and evaluation (step 5)

`train(build_training_rows(...), artifact_dir)` on the sample. It takes about 0.35 s, writes `model.pkl` and `model.json` and returns the report.

| Split | Value |
|---|---|
| split_time | 2026-01-29T03:00:00Z (70 % point of 714 eligible rows) |
| train rows / positives | 473 / 25 |
| test rows / positives | 215 / 20 |
| purged (36 h gap) / grouped out / relabelled at split | 26 / 0 / 0 |
| censored rows ignored | 6 |
| model_version | `logreg-7170fa340d6fa4bd` (pickled scikit-learn pipeline; the earlier versions `logreg-bec4ef1481f681cf` and `logreg-1121738bccbce059` had identical metrics) |

| Test scorer | Average precision | Brier |
|---|---|---|
| logistic regression | 1.000 | 0.000565 |
| threshold rule 8 °C | 0.456 | 0.0531 |
| prevalence baseline | 0.093 | 0.0860 |

With only 20 test positives these numbers are indicative, not precise.

| Slice (logistic regression) | Rows / positives | AP | Brier |
|---|---|---|---|
| carrier central | 72 / 6 | 1.000 | 0.000444 |
| carrier coast | 71 / 9 | 1.000 | 0.000646 |
| carrier north | 72 / 5 | 1.000 | 0.000607 |
| latest reading ≤ 1 h old | 185 / 16 | 1.000 | 0.000436 |
| latest reading > 1 h old | 30 / 4 | 1.000 | 0.001361 |

| Sanity check | Result |
|---|---|
| Trained on shuffled labels (20 seeds), tested on true labels: `python tools/shuffled_label_check.py` | median AP 0.088 (min 0.050, max 0.838), chance level 0.093: the evaluation is sound |
| Generator variant, warming 0.3 °C/h instead of 0.9 | AP 0.867, Brier 0.0307: metrics drop honestly |
| `--seed 7 --size 60` | 95 train / 53 test rows, trains normally |
| `--seed 1729 --size 10` | stops with `insufficient_training_data` (0 mature train rows) |

The shuffled-label median comes from the assistant's sandbox with an equivalent plain-Python fit; the script runs the same check with the pinned scikit-learn pipeline. The other three rows are development checks, run once in scratch folders (never `data/`); they are not scripted.

## End-to-end demo (`python tools/demo.py`)

Actual output of `python tools/demo.py` in a fresh process on my machine (Python 3.12.8, scikit-learn 1.8.0). The assistant's sandbox (Python 3.12.3) produced byte-identical output:

```
[1] Data
    4402 events, 47 labels, 720 decisions; observation_end 2026-02-11T12:00:00Z; lag 30 h
[2] Point-in-time training rows (immature label windows are censored, not dropped)
    {"censored": {"decision_after_cutoff": 0, "label_window_immature": 6}, "duplicates_merged": 0, "eligible": 714, "input": 720, "positive": 47, "rows": 720}
    events: {"accepted": 3922, "conflict": 0, "duplicate": 480, "input": 4402, "received_after_cutoff": 0, "superseded_on_arrival": 0}
    censored: e-00237 2026-02-10T02:00:00Z label_window_immature (label mature at 2026-02-11T14:00:00Z)
    censored: e-00238 2026-02-10T03:00:00Z label_window_immature (label mature at 2026-02-11T15:00:00Z)
    censored: e-00239 2026-02-10T04:00:00Z label_window_immature (label mature at 2026-02-11T16:00:00Z)
[3] Train: purged temporal split, logistic regression vs baselines
    split_time 2026-01-29T03:00:00Z: train 473 (25 pos), test 215 (20 pos), purged 26
    logistic_regression    AP 1.0  Brier 0.000565
    prevalence_baseline    AP 0.093023  Brier 0.085983
    threshold_rule_8c      AP 0.455814  Brier 0.05311
    model_version logreg-7170fa340d6fa4bd
[4] Engine replay in delivery order + offline/online parity
    ingest True 3922, False 480; parity 720/720 decisions; stats {"accepted": 3922, "duplicate": 480, "scores": 720}
[5] Demonstrations
    duplicate delivery -> ingest returned False
    correction e-00000-8 r2 received 2026-01-01T19:00:00Z after decision 2026-01-01T14:00:00Z: decision unchanged = True
    reversed delivery (rev 2 delivered first, rev 1 received earlier) -> rev 1 ingest returned True
    reversed receipt (rev 1 received after rev 2) -> rev 1 ingest returned False
    eviction (max_entities=20): resident 20, evicted 220, eviction list remembers 20
    late correction for recently evicted shipment e-00219 -> rejected: entity_evicted
    late correction for evicted long ago shipment e-00000 -> ACCEPTED as a new shipment (known limitation: the eviction list is bounded)
    failed reload -> returned False, last_error artifact_corrupt, previous model still serving identical scores = True
[6] Uninterrupted replay vs snapshot/restore replay
    max_entities=10000: identical final snapshot True, identical decision outcomes True
    max_entities=20: identical final snapshot True, identical decision outcomes True
Done. Outputs in demo_output/
```

| README demonstration | Result |
|---|---|
| Duplicate delivery | `ingest` returned False |
| Correction arriving after a decision | the 14:00 decision is byte-identical before and after the correction (received 19:00) |
| Reversed revision order | rev 2 delivered first with rev 1 received earlier → True; rev 1 received after rev 2 → False |
| Immature label window | 6 decisions censored as `label_window_immature`, with the time each label becomes mature |
| Eviction | `max_entities=20`: 20 resident, 220 evicted; a late correction for a recently evicted shipment is rejected (`entity_evicted`); one for a long-evicted shipment is accepted as new (known limitation) |
| Failed reload | returned False, `last_error` `artifact_corrupt`, previous model kept serving identical scores |
| Uninterrupted vs snapshot/restore replay | identical final snapshot bytes and identical outcomes for all 720 decisions, at `max_entities` 10000 and 20 |
| Offline/online parity | 720 of 720 decisions have the same feature digest and probability |


`model_version` `logreg-7170fa340d6fa4bd` is identical on both machines. Most of the demo's run time is scikit-learn scoring.
