# Decision Record

## Customer interpretation and rejected notes

### Note 1: "Use the latest corrected value when replaying every old decision." Not adopted (step 2)

- **What we do instead:** a replay uses what was known at the decision time. A correction counts only from its own receipt time (`received_at ≤ as_of`). Test: `test_correction_after_a_decision_does_not_rewrite_it`.
- **Why:** replaying with later corrections creates features production never had (train/serve skew, inflated offline metrics). It also makes the same historical question return different answers over time, which breaks "reproducible historical decisions".
- **Customer impact:** replays match what operators were shown at the time, so they stay auditable. If the customer wants "what we would have said with perfect data", that can be a separately labelled restatement report, never mixed into replay or training. Offline metrics may look slightly lower, but they will match production.

### Note 2: "Randomly split decision rows; the same shipment may appear in both sets." Not adopted (step 5)

- **What we do instead:** a temporal split. The model is trained on the past and tested on the future, with a purge gap and no shipment on both sides (details under "Model, split and evaluation").
- **Why:**
  - A random split lets the model test on rows whose neighbours it trained on. The 11:00 and 14:00 decisions of one shipment share most readings, and their 6-hour label windows overlap.
  - It also trains on the future to predict the past.
  - Both make offline metrics look better than production will be.
- **Customer impact:** the reported numbers are lower but honest, so there is no surprise drop after launch. With only 20 test positives the numbers are also noisy, which is stated next to every result.

### Note 3: "If model loading fails, report zero risk so customer dashboards remain green." Not adopted (engine)

- **What we do instead:**
  - At startup, `RiskEngine(...)` loads and validates the artifact (format, feature schema, scikit-learn version, SHA-256 of the pickle before unpickling). With no valid model it raises `ArtifactError` and does not start.
  - `reload_model` validates the new artifact *before* switching. If loading fails it returns `False`, the previous model keeps serving, and `stats()["last_reload_error"]` records why.
- **Why:** in cold chain, a green dashboard during an outage can mean spoiled product shipped with a clean record. A fake "0 % risk" is the most dangerous possible failure, and it breaks the brief's rule against fabricating successful results.
- **Customer impact:** a model problem appears as a startup failure or a reload alert that on-call must handle, never as a false all-clear.
- **Tests:** `test_note3_no_valid_model_means_no_engine`, `test_note3_failed_reload_keeps_the_previous_model`. The demo also shows a failed reload.

### Note 4: "Keep every event forever; enforce memory only on the prediction cache." Not adopted (engine)

- **What we do instead:** memory is bounded where it grows:
  - at most `max_entities` resident shipments, evicting the least recently ingested
  - at most `MAX_REVISIONS_PER_SHIPMENT` (1,000) stored revisions per shipment
  - a bounded list of evicted shipment IDs

  There is no prediction cache at all, because scoring takes about 1.5 ms.
- **Why:** "every event forever" in a long-running service grows without limit and ends in an out-of-memory crash, an outage for everyone. The README also says the limit applies to resident entities and their indexes, not only outputs.
- **Customer impact:**
  - Questions about evicted shipments are rejected by the engine (`entity_evicted`) rather than answered from partial history; they can be answered offline by replaying the event export with `build_training_rows`.
  - `max_entities` should be sized above the number of concurrently active shipments.
- **Tests:** `test_note4_entity_bound_and_late_correction_after_eviction`, `test_note4_per_shipment_history_cap`, and the sample replay with `max_entities=20`.

### Interpretation: `Result.entity_id`

The README calls it "the invoice ID for reconciliation; query ID for search". That is a leftover from other variants of the same take-home template; the generator also contains invoice and search modes. Here `entity_id` is the **shipment ID** passed to `score()`.

## Time, revisions, and input validation

### Timestamps (step 1)

- **Accepted input format:** `YYYY-MM-DDTHH:MM:SS[.f{1,6}](Z|+HH:MM|-HH:MM)`, uppercase `T` and `Z` (the RFC 3339 profile of ISO 8601).
- **Parser:** an explicit regular expression plus `datetime(...)`, not `datetime.fromisoformat`. Checked on Python 3.12: `fromisoformat` returns a *naive* datetime for `2026-01-01`, accepts ISO basic (`20260101T000000Z`) and week dates (`2026-W01-1T00:00:00Z`), and silently truncates `…00.1234567Z`. Its accepted formats also changed in 3.11. An explicit grammar behaves the same on every supported Python version and gives a precise error code.
- **ASCII digits only:** `[0-9]` instead of `\d`, because `\d` also matches non-ASCII digits and `int("٢٠٢٦") == 2026`.
- **UTC on entry:** every timestamp is converted to UTC immediately, so equivalent offsets are equal values (`2026-01-01T05:30:00+05:30` == `2026-01-01T00:00:00Z`). `-00:00` is read as UTC, since the instant is still known.
- **Missing offset is rejected** (`missing_utc_offset`). UTC is never assumed: a local-time string would shift every reading by hours without anyone noticing.
- **Precision:** at most 6 fractional digits. More would need silent rounding, which changes the instant.
- **Range:** years 1900–9998 (`timestamp_out_of_range`). This only keeps date arithmetic (`as_of + 6h + 30h`, look-back windows) from overflowing. Plausible-but-wrong device clocks are handled by the clock policy (step 2), not rejected here.
- **Datetime arguments** (`as_of`, `observation_end`) must be `datetime` objects with a UTC offset. Naive datetimes raise `naive_datetime`; dates, strings and other types raise `not_a_datetime`. Values are converted to UTC.
- **Output format:** always `YYYY-MM-DDTHH:MM:SSZ`, plus `.ffffff` only when microseconds are non-zero, so one instant always produces one string. Built with f-strings, not `strftime`, whose padding is platform-dependent.

### Record validation (step 1)

| Field | Rule | Error code |
|---|---|---|
| any record | must be a mapping with exactly the documented fields | `not_a_mapping`, `missing_field`, `unknown_field` |
| `entity_id`, `event_id`, `incident_id`, `carrier`, `kind` | string, 1–128 characters, no surrounding whitespace, printable | `invalid_string` |
| `revision` | exact `int` ≥ 1 (`True`, `1.0`, `"1"` rejected) | `invalid_revision` |
| `value` | exact `int` or `float`, finite, ≥ −273.15 °C; stored as `float`, `-0.0` normalized to `0.0` | `invalid_number`, `non_finite_value`, `below_absolute_zero` |
| `kind` | must equal `temperature_c` | `unsupported_kind` |
| all timestamps | as above | `invalid_timestamp`, `missing_utc_offset`, `timestamp_out_of_range` |
| label | `label_available_at` ≥ `incident_at` | `label_before_incident` |

- **Unknown fields are rejected, not ignored.** A new field such as `"unit": "F"` could change what `value` means; ignoring it would misread data silently.
- **Carrier is not restricted** to the values seen in the sample, because that would be hard-coding. Unseen carriers are handled at feature time (step 3).
- **A future device clock** (`event_at` later than `received_at`) is *valid* at this layer. The reading may be fine; only its device time is wrong. The policy and its counter come in step 2.
- **Inputs are never mutated.** Validators only read the caller's mapping and return a frozen dataclass holding a normalized copy. "Same content" for duplicate/conflict detection (step 2) compares these normalized objects, never raw strings.

### Revisions, duplicates, conflicts, device clocks (step 2)

Implemented in `_store.py`: an `EventStore` holding one `EntityHistory` per shipment, which stores every revision of every reading.

- **Identity** is `(entity_id, event_id, revision)`. The same `event_id` in two shipments means two different readings.
- **Visibility:** a revision is known at T iff `received_at ≤ T` (inclusive). This mirrors the label window `(t, t+6h]`: a decision at t knows everything received up to and including t, and predicts strictly after t. 462 sample readings arrive exactly at a decision time, so this choice affects features.
- **Winner:** among the revisions known at T, the highest revision wins regardless of delivery order. A lower revision delivered later never overrides a higher one.
- **Add outcomes:**

| Situation | Status | `changed` (what `ingest` will return) |
|---|---|---|
| New reading, or a new revision that wins at some time | `accepted` | `True` |
| A higher revision had already been received by this one's receipt time | `superseded_on_arrival` (stored, never wins) | `False` |
| Same identity, same normalized content | `duplicate` (not stored again) | `False` |
| Same identity, different content | `ConflictError`; the first version is kept | raises |

- **Why the `changed` rule is exact.** As T grows, the winning revision can only stay the same or get higher. A new revision r received at time t_r therefore changes some answer iff it wins at t_r, that is, iff no higher revision had been received by t_r. An oracle test compares every as-of answer before and after each add. It covers all 120 delivery orders of a hand-built tricky set and every event in the sample.
- **Content comparison** uses the normalized `Event` from step 1 (all eight fields, UTC instants, float values). A redelivery written with `+05:30` is therefore a duplicate. A redelivery with a different `received_at` is a conflict, because receipt time decides availability. Considered and rejected: treating receipt-time-only differences as duplicates and keeping the earliest.
- **Device clocks:** a correct clock cannot measure after receipt. So if `event_at` is more than `CLOCK_SKEW_TOLERANCE` (5 minutes, allowing normal drift) after `received_at`, the reading is stored but not used for features, and is counted as `future_device_clock`, which becomes a score reason.
  - The revision is resolved *before* the clock check. If the newest known revision has a bad clock, the reading is unusable; there is no fallback to the value it replaced. A correction that fixes the clock makes the reading usable from that correction's receipt.
  - Device clocks far in the past need no separate rule: they fall outside the feature look-back window (step 3).
  - Considered and rejected: clamping `event_at` to `received_at` (keeps the value but invents a time) and rejecting the record at ingest (loses the record and the evidence of a faulty device).
- **Determinism:** query results are sorted by `(event_at, event_id)` and stored events by identity. On the sample, three delivery orders (file, reversed, seeded shuffle) give identical stored state and identical as-of answers for all 720 decisions.

### Error types and codes

Every deliberate error carries a stable `.code` (for counting in `stats()`) and a human-readable `.detail`.

| Exception | Raised when |
|---|---|
| `InvalidRecordError(ShipmentRiskError, ValueError)` | an event, label, decision or JSONL line is malformed |
| `InvalidArgumentError(ShipmentRiskError, ValueError)` | a method argument is malformed |
| `ConflictError(ShipmentRiskError, ValueError)` | same identity, different content: code `conflict` for events (first version kept), `label_conflict` for labels (build stops) |

JSONL-level codes (`_io.read_jsonl`, always with `file:line`):

| Code | Meaning |
|---|---|
| `invalid_json` | the line is not valid JSON, or nesting is too deep |
| `duplicate_json_key` | the same key appears twice in one object |
| `non_finite_value` | `NaN`, `Infinity` or `-Infinity` literal |
| `blank_line` | an empty line inside the file |
| `not_a_mapping` | the line is valid JSON but not an object |
| `invalid_encoding` | the file is not valid UTF-8 |

A missing file raises the standard `FileNotFoundError`: an environment problem, not malformed data.

Other input codes (step 4):

| Code | Raised by | Meaning |
|---|---|---|
| `not_iterable` | `build_training_rows` | `events`, `labels` or `decisions` is not a list/iterable of records |
| `unsupported_schema_version` | `read_manifest` | `MANIFEST.json` schema is not 1 |
| `invalid_label_lag` | `read_manifest` | `label_reporting_lag_hours` is not a positive integer |
| `invalid_manifest` | `read_manifest` | `files` is not an object |
| `label_lag_mismatch` | `check_reporting_lag` | the manifest's lag differs from the code's 30 h |

Training and artifact codes (step 5):

| Code | Exception | Meaning |
|---|---|---|
| `insufficient_training_data` | `InvalidArgumentError` | after the split there is no positive, no negative, or no test row. Nothing is written |
| `invalid_training_row` | `InvalidRecordError` | a row is malformed: missing fields, the wrong feature names, a label that is not 0/1, or a NaN/infinite feature |
| `feature_schema_mismatch` | `InvalidRecordError` (rows) / `ArtifactError` (artifact) | built with different feature definitions than the code computes |
| `not_a_path` | `InvalidArgumentError` | `artifact_dir` is not a `str` or `Path` |
| `artifact_missing` | `ArtifactError` | `model.json` (metadata) or `model.pkl` (pipeline) is missing |
| `artifact_corrupt` | `ArtifactError` | metadata is not valid JSON, `model.pkl` does not match its recorded SHA-256, or it cannot be unpickled |
| `artifact_format` | `ArtifactError` | unknown format version, a different scikit-learn version than the one that wrote it, or the pickle is not a scikit-learn `Pipeline` |

Engine codes:

| Code | Exception | Raised by | Meaning |
|---|---|---|---|
| `invalid_max_entities` | `InvalidArgumentError` | `RiskEngine()` | not a positive int (`True` and `2.5` rejected) |
| `invalid_entity_id` | `InvalidArgumentError` | `score` | empty, surrounding whitespace, or not a string |
| `not_a_path` | `InvalidArgumentError` | constructor, `snapshot`, `restore`, `reload_model` | a path argument is not `str`/`Path` |
| `entity_evicted` | `RetentionError` | `ingest`, `score` | the shipment was evicted; its history is gone |
| `history_full` | `RetentionError` | `ingest` | the shipment already stores 1,000 revisions |
| `invalid_snapshot` | `SnapshotError` | `restore` | missing, unreadable, wrong format, or too many shipments |

`reload_model` never raises for a bad artifact. It returns `False` and sets `stats()["last_reload_error"]` to the `ArtifactError` code, or `artifact_unreadable`.

Score reason added by the engine: `unknown_shipment` (nothing was ever ingested for this shipment).

Censor reasons (`row.metadata.censor_reason`, step 4): `label_window_immature` and `decision_after_cutoff`, as described under "Training rows, labels and censoring".

Data-quality reason codes (step 3). These are not errors; `score()` reports them in `reasons`:

| Code | Meaning |
|---|---|
| `future_device_clock` | at least one known reading was excluded because its device time is more than 5 minutes after its receipt |
| `no_readings_in_window` | no usable reading in the 6-hour look-back window, so the temperature features are `null` |

## Algorithm, evaluation, and baseline

### Features (step 3)

A single function, `compute_features(KnownReadings)` in `_features.py`, is used both by training (step 4) and by live scoring (`RiskEngine.score`). Both paths call `EventStore.known_at()` and then this function, so offline and online features agree by construction. The `feature_digest` lets tests prove it.

| Feature | Definition | Why |
|---|---|---|
| `latest_temp_c` | value of the most recent usable reading in the window | current state |
| `max_temp_c` | highest value in the window | a recent spike |
| `mean_temp_c` | mean of the window (`math.fsum`) | level, less noisy than one reading |
| `trend_c_per_hour` | least-squares slope of value over device time; needs 2+ distinct times | excursions are preceded by warming |
| `reading_count` | usable readings in the window | data coverage |
| `hours_since_latest_reading` | `as_of` minus the latest device time, clamped to 0–6 | staleness; a silent sensor is a risk signal |
| `carrier` | carrier on the latest usable reading (raw string) | carriers differ in risk |

- **Look-back window:** device time in `(as_of − 6h, as_of + 5 min]`. Six hours matches the prediction horizon; the upper end is the store's clock tolerance. Ages are clamped at 0 for tolerated skew. Device clocks far in the past fall outside the window, so they need no separate rule.
- **Missing facts are `null`, never invented numbers.** For example, there is no latest temperature when no reading is in the window. The model decides how to fill them and records that in its artifact (step 5).
- **Carrier stays a raw string.** It is encoded by the model (step 5), so the carrier list is not hard-coded and unseen carriers do not crash.
- **Deliberately excluded:**
  - *Shipment age* and *decision hour*. In this synthetic data only one decision offset is ever positive, so these features would memorize the generator's schedule instead of learning from temperatures, and would break on data with other decision times.
  - Features derived from receipt lag.
- **Numerics:** sums use `math.fsum`, `-0.0` is normalized, and every value is canonical-JSON safe (no NaN/inf).
- **Digest:** `sha256` of canonical JSON of `{"schema": "features-v1", "features": values}`. Bumping `FEATURE_SCHEMA_VERSION` changes every digest, so an artifact trained on old definitions can be detected (step 5).
- **Evidence (sample):**
  - Availability by `event_at` would put the backdated 19.0 °C reading into the features of 94 of the 720 decisions, and change the features of 645. The receipt-time rule gives 0.
  - For every decision, features from the full store are byte-identical to features from a store fed only the records received by `as_of`.

### Training rows, labels and censoring (step 4)

`build_training_rows` returns `build_dataset(...).rows` (`_dataset.py`). The same call also produces a **report** of counts, used by the demo and RESULTS.md, so that nothing disappears silently. The label rules live only in `_labels.py`; step 5 reuses them with the split time as the cutoff.

- **One row per unique decision**, sorted by `(as_of, entity_id)`. Rows are plain JSON:
  - `entity_id`
  - `as_of` (canonical UTC string)
  - `features` (dict, or `null` if unknown)
  - `label` (`1`, `0`, or `null` if censored)
  - `metadata`: `eligible`, `censor_reason`, `incidents`, `feature_digest`, `feature_schema`, `data_reasons`, `label_window_end`, `label_mature_at`, `observation_end`, `duplicate_decisions`
- **Label:** `1` iff an incident falls in `(as_of, as_of + 6h]` and its label was available by the cutoff. The window is open at `as_of` (an incident already happening is not a future risk) and closed at +6h, as the brief states.
- **Maturity:** `as_of + 6h + 30h ≤ cutoff`, applied to positives *and* negatives. Keeping immature positives while dropping immature negatives would inflate the positive rate. Censored rows are **returned with `label: null` and a `censor_reason`**, never dropped silently. They keep their features, because those were known at `as_of`; only the outcome is unknown.
- **Censor reasons:**
  - `label_window_immature`: the outcome might not be fully reported yet.
  - `decision_after_cutoff`: `as_of` is after the export's knowledge cutoff, so features and label are both `null` rather than computed from incomplete data.
- **Records after the cutoff** (events received later, labels available later) are ignored and counted. They did not exist yet for this export.
- **Conflicts:**
  - Event conflicts keep the first delivered version and are counted, exactly as the live engine will, so offline rows equal what the engine would compute.
  - Label conflicts stop the build (`label_conflict`). There is no live path to mirror, and two different incident times for one incident cannot be resolved automatically.
- **Malformed input** anywhere stops the build with its position (e.g. `events[17]: missing_utc_offset`). A broken export is a pipeline bug to fix, not something to train around.
- **Duplicates:** identical decisions (also when written with other offsets) become one row (`duplicate_decisions`); identical labels are counted.
- **Reporting lag:** `LABEL_REPORTING_LAG` (30 h) is a constant in `_labels.py`, because the public signature has no lag parameter. It is checked against `MANIFEST.json` with `check_reporting_lag`, and a mismatch stops training instead of silently using the wrong rule.
- **Late-label signal:** labels that took longer than 30 h are counted as `beyond_reporting_lag`. A non-zero count means the maturity rule may be hiding other late positives, so it should be monitored.
- **For step 5:** `metadata.incidents` keeps `label_available_at`, so the temporal split can re-check availability at its own, earlier cutoff.
- **Evidence:**
  - On the sample: 720 rows, 714 eligible, 47 positive, 6 censored (all `label_window_immature`).
  - Labels and eligibility match an independent brute-force check for every row.
  - Shuffled or reversed inputs give byte-identical rows.
  - Generator variants (run into scratch folders, never `data/`): lag 48 stops with `label_lag_mismatch`; an earlier cutoff gives 2 `decision_after_cutoff`, 10 late events and 3 late labels ignored; slower labels give 10 `beyond_reporting_lag`; duplicated decisions give 240 merged; an extra 17 h decision and an earlier incident move positives to other decision times (93 and 94) with no code change.

### Model, split and evaluation (step 5)

`train(rows, artifact_dir)` (`_training.py`) writes `model.pkl` (the fitted pipeline) and `model.json` (its metadata) and returns the evaluation report.

**Split** (purged, temporal, shipment-grouped):

1. Keep the eligible rows (step 4) and sort them by time. Censored rows are counted and never used.
2. `split_time` is the `as_of` of the row at the 70 % point. Test rows are those with `as_of ≥ split_time`.
3. Train rows must be *mature at split_time* (`as_of + 6h + 30h ≤ split_time`). They are re-labelled using only labels available by `split_time`, applying the same `_labels.py` rules with the split as the cutoff. Rows in the 36-hour gap are **purged**, because their label windows or reporting delays reach into the test period.
4. No shipment appears on both sides.

On the sample: split at 2026-01-29T03:00Z, 473 train rows (25 positive), 215 test rows (20 positive), 26 purged, 0 grouped out, 0 relabelled.

**Model:** a scikit-learn `Pipeline`, saved with pickle (`_model.py`).

| Pipeline step | What it does |
|---|---|
| `SimpleImputer(strategy="mean")` → `StandardScaler` | a missing number becomes the training mean (so it contributes 0 after scaling); every numeric feature is standardized; a constant feature gets scale 1 |
| `OneHotEncoder(handle_unknown="ignore")` | one column per carrier seen in training; an unseen or missing carrier gets all zeros (no adjustment) |
| `LogisticRegression(C=1.0)` (lbfgs) | log-loss plus a mild L2 penalty, ½·‖w‖², on the weights only (not the intercept). A test checks the optimality condition (gradient ≈ 0) of exactly that objective. |

- **All statistics are learned from training rows only.** `fit_logistic` never sees test rows, so the scaler and imputer cannot leak test information.
- **The artifact is two files:**
  - `model.pkl` is the pickled pipeline.
  - `model.json` is readable metadata: format, feature schema, the scikit-learn version, the SHA-256 of `model.pkl`, a human-readable copy of all fitted parameters, the training info, and `model_version`.
- **`model_version` comes from the fitted parameters,** rounded (fill values, scaler means and scales, carriers, coefficients, intercept) plus the training info. It does *not* come from the pickle's bytes, because pickle bytes are not stable: a single load/save round trip changed them in a test. Coefficients are rounded to 6 decimals inside the pipeline, so every machine scores with exactly the numbers the version describes.
- **Dependencies are pinned exactly** in `pyproject.toml` (`scikit-learn==1.8.0`, `numpy==2.4.4`, `scipy==1.17.1`). A pickle only loads reliably with the scikit-learn version that wrote it, so the loader refuses a different version (`artifact_format`), and upgrading the library means retraining.
- **Cost:** training takes about 0.35 s. A single live score takes about 1.5 ms, against about 0.05 ms for the earlier hand-written scoring, because each call goes through scikit-learn's input checks. That is fine for this workload.
- **History of this choice:**
  1. Fitted first with Newton's method in plain Python (`logreg-bec4ef1481f681cf`).
  2. Then with scikit-learn, exporting the weights to JSON (`logreg-1121738bccbce059`).
  3. Finally as a pickled pipeline, at my request, to use the standard preprocessing classes (`logreg-7170fa340d6fa4bd`).

  All three give identical metrics, slices and sanity checks to 6 decimals; no feature is ever missing in the training rows, so imputation-then-scaling equals the earlier hand-written encoding.

**Baselines:**

- `prevalence_baseline`: always the training positive rate ("no skill").
- `threshold_rule_8c`: latest reading ≥ 8 °C (the usual 2–8 °C cold-chain limit), turned into the training positive rate on each side of the threshold.

**Metrics:** average precision (ranking; appropriate for ~9 % positives) and Brier score (probability quality). With only 20 test positives, moving one positive changes these noticeably, so they are reported as indicative, not precise. Confidence intervals (e.g. a shipment bootstrap) were considered and left out to keep the training step small.

**Slices:** carrier, and data freshness (latest reading ≤ 1 h old vs older).

**Results on the sample (test rows):**

| Scorer | Average precision | Brier |
|---|---|---|
| logistic regression | 1.000 | 0.000565 |
| threshold rule 8 °C | 0.456 | 0.0531 |
| prevalence baseline | 0.093 | 0.0860 |

**AP 1.0 looks like leakage, so it was checked:**

- Training on *shuffled* labels (20 seeds, `tools/shuffled_label_check.py`) gives a median test AP of 0.088, at the chance level (positive rate 0.093). So the score comes from what the model learns, not from the evaluation. Feature leakage is ruled out separately: receipt-time filtering keeps the backdated 19.0 °C readings out of every decision.
- A generator variant with a weaker warming signal (0.3 °C/h instead of 0.9) drops AP to 0.867 (development check, run once; not scripted).
- Together with step 3's proof (features byte-identical when only records received by `as_of` are known), the perfect score reflects how strong the synthetic signal is: bad shipments warm about 4.5 °C before the incident, against 0.6 °C noise. Real data will not be this clean. Accuracy is not the acceptance criterion; the baselines and slices are reported for honesty, not as a target.

**Model choice:** tree ensembles (random forest, gradient boosting) were considered and deferred:

- With 20 test positives, any improvement over a perfect-AP linear model cannot be measured.
- Their probabilities usually need separate calibration.
- They would pickle just as easily, so the artifact is not the obstacle; the reason is the 20-positive test set.

The planned path: run them in shadow mode once real data has enough labeled incidents.

### Online scoring (engine prototype)

`RiskEngine` reuses the offline path exactly: `ingest` = `validate_event` → `EventStore.add`; `score` = `known_at` → `compute_features` → `model.predict`.

| Payload field | Meaning |
|---|---|
| `probability` | model probability rounded to 6 decimals (finite, in [0, 1]) |
| `model_version` | `logreg-<16 hex>` of the artifact used |
| `feature_digest` | sha256 of the features and their schema version |
| `degraded` | `true` iff `reasons` is non-empty |
| `reasons` | sorted codes: `no_readings_in_window`, `future_device_clock`, `unknown_shipment` |
| `features`, `feature_schema` (extra) | the exact model inputs, so any score can be traced |

- **Parity:** after replaying all 4,402 events in delivery order, all 720 sample decisions have the same feature digest and probability online as offline.
- **Unknown shipment:** returns a degraded result rather than an error. This matches the offline row for a decision with no data, and `degraded: true` means "no reliable answer".

## Idempotency, persistence, and failure recovery

- **Duplicates are idempotent (step 2):** an identical copy is not stored again, returns `changed=False`, and leaves every as-of answer unchanged. A conflict never overwrites the stored version.
- **Offline duplicates (step 4):** repeated decisions become one row, identical labels are counted once, and the order of inputs does not change the rows.

- **`ingest` (engine):** returns `True` only if an as-of answer can change. Duplicates and never-winning revisions return `False`. Invalid records (`InvalidRecordError`), conflicts (`ConflictError`) and retention violations (`RetentionError`) raise and are counted.
- **Snapshot (prototype):** `snapshot(path)` writes canonical JSON containing:
  - the format and `max_entities`
  - resident shipments in eviction order, each with all stored revisions sorted by identity
  - the evicted-ID list and the counters

  It is **deterministic** (same state → same bytes; no clock time) and **atomic** (temporary file, `fsync`, `os.replace`, from step 5's `write_atomic`).
- **Restore (prototype):** `RiskEngine.restore(artifact_dir, snapshot)` loads the model from `artifact_dir` and **re-ingests every record through the normal validation**, then restores the evicted list and counters. A bad file raises `SnapshotError("invalid_snapshot")`; a bad model raises `ArtifactError`.
- **Replay comparison (README):** uninterrupted replay vs first half → snapshot → restore → second half gives identical final snapshot bytes and identical outcomes for all 720 decisions, with `max_entities` 10000 and with 20 (220 evictions). Run by `tools/demo.py` and tested in `test_snapshot_restore_replay_equals_uninterrupted_replay`.
- **Failed reload:** returns `False`; the old model keeps serving identical scores (note 3).

## Memory bounds, eviction, and concurrency

- **Entity bound:** at most `max_entities` resident shipments.
  - A new shipment at the limit evicts the **least recently ingested** one.
  - Recency changes on ingest only, never on `score()`, so query traffic cannot change state and replays evict identically.
- **Late correction after eviction (README question):** rejected with `RetentionError("entity_evicted")` and counted. Re-creating the shipment from one late record would give wrong features. `score()` for an evicted shipment raises the same error.
- **The evicted-ID list is bounded too** (at most `max_entities`, oldest dropped first). Once a shipment drops off that list, a very late record for it is accepted as a *new* shipment. The demo shows this deliberately; it is listed under remaining work.
- **Per-shipment history:** complete history is kept up to 1,000 stored revisions; beyond that, records are rejected (`history_full`). History is never partially dropped, so historical questions about resident shipments are answered exactly. Supported queries: any `as_of` for a resident shipment.
- **Memory:** about `max_entities × 1,000` events, plus `max_entities` evicted IDs, plus one ordered index entry per resident shipment. The sample replay with `max_entities=20` never exceeds 20 resident shipments.
- **Concurrency:** one lock guards all state.
  - `score` takes a consistent view (immutable readings plus the `(model, version)` pair) under the lock and computes outside it.
  - `reload_model` loads and validates outside the lock, then swaps the pair in one assignment.
  - Test: 3 scoring threads during 30 reloads (models A, B and a broken artifact). Every score's probability matched its reported model version, so no score ever mixed two models.

## Security and operational monitoring

Step 1 (input handling):

- All input is treated as untrusted. Identifiers are capped at 128 characters and timestamps at 40. Excessively deep JSON nesting is rejected as `invalid_json` instead of crashing the process.
- Strict JSON: `NaN`/`Infinity` literals and duplicate keys are rejected. Python's `json` accepts both by default, and with duplicate keys it would silently keep the last value.
- Invalid UTF-8 is rejected with an explicit code.
- Error codes are stable strings so they can be counted and alerted on (`stats()` counters).
- SHA-256 is used only as a fingerprint (feature digest, model version, pickle integrity), not for secrecy.
- **Model artifact (pickle):**
  - **Risk:** unpickling a file can execute arbitrary code.
  - **Mitigations:** the loader checks the metadata, the feature schema and the scikit-learn version, then the SHA-256 of `model.pkl` against `model.json`, all **before** unpickling. So a swapped or corrupted file is refused without being loaded.
  - **Limits:** this is an integrity check, not authentication: someone who can write both files can replace both. Artifacts must come only from our own training job and a write-protected location. Signing artifacts (e.g. HMAC) or returning to a JSON-only artifact would remove this risk; see remaining work.
  - Any failure raises `ArtifactError`; nothing half-loaded is ever used.
  - Files are written atomically (temporary file, `fsync`, `os.replace`), so a crash never leaves half a file.

- **Monitoring (engine):** `stats()` returns `resident_entities`, `max_entities`, `evicted_remembered`, `model_version`, `last_reload_error`, and counters: `accepted`, `duplicate`, `superseded_on_arrival`, `conflict`, `rejected_invalid`, `rejected_entity_evicted`, `rejected_history_full`, `evicted`, `scores`, `degraded_scores`, `reload_succeeded`, `reload_failed`.
  - Suggested alerts: any `reload_failed`; any `conflict`; rising `rejected_*` or `degraded_scores / scores`; `rejected_entity_evicted` (active shipments are being evicted, so raise `max_entities`).
- **Snapshots** are canonical JSON written atomically. Restore re-validates every record, but the prototype does **not** checksum snapshot files (remaining work).

## Launch criteria and rollback

**Launch criteria** (all must hold before scores drive any action):

1. The test suite passes on the production Python version, and `python tools/demo.py` reproduces the RESULTS.md numbers and `model_version`.
2. **Shadow mode for at least 2 weeks** on the live feed: scores are logged next to the existing 8 °C alarms; no actions are taken from them.
3. A daily parity check: replaying the day's events offline gives the same feature digests as the online scores (100 %).
4. On matured shadow data with enough incidents to be meaningful, the model beats the 8 °C threshold rule on both average precision and Brier (confirmed with confidence intervals at that point), and no carrier slice is worse than the rule.
5. Operations are clean: no unexplained conflicts, degraded rate below an agreed threshold, no reload failures, memory within the bounds.
6. Remaining-work items 1–3 (snapshot checksum, late-data flag, eviction memory) are done.

**Rollout:**
1. Shadow.
2. Advisory: the probability is shown beside the existing alarms, and `degraded` is shown as "no reliable answer".
3. Alerting for one carrier.
4. All carriers.

**Rollback:**
- Every artifact is kept in its own directory, versioned by content hash. `reload_model(previous_dir)` switches back without a restart, and a failed reload never takes the old model down.
- If the engine itself misbehaves, hide the scores. The existing threshold alarms stay on throughout and remain the safety net.
- State can be rebuilt from the last snapshot, or by replaying the event log.

## Known limitations and time spent

Step 1 limitations:

- Strict schema: if the customer adds a field, those records are rejected until the code is updated. This is deliberate, because silent misreading is worse.
- RFC 3339 also allows lowercase `t`/`z`; we accept only uppercase.
- Identifiers are compared exactly as given; there is no Unicode normalization (a composed and a decomposed `é` are different IDs).
- `value` is checked only for physical impossibility (below absolute zero). Sensor sentinel codes such as `999` are not detected.
- `read_jsonl` loads a whole file into memory. That is fine at this size; very large files would need a streaming reader.
- So far, tests have run on Python 3.12 only, not on 3.11 (the declared minimum).

Step 2 limitations:

- The store itself is unbounded; the engine enforces the entity cap and the per-shipment cap (offline stays unbounded).
- `superseded_on_arrival` revisions are kept, because conflict detection and complete snapshots need them. They cost memory until retention prunes them.
- `CLOCK_SKEW_TOLERANCE` = 5 minutes is a judgement call. The sample contains no skew to calibrate it.
- Conflicts are strict, including differences in receipt time only.
- `known_at` scans every reading of the shipment, which is fine for bounded histories.

Step 3 limitations:

- The features are deliberately simple, built from temperature only. There is no ambient temperature, door-open, route or product data.
- The 6-hour window is a judgement call, not tuned.
- With 2–3 readings, the trend is noisy. There is no outlier handling, so a single spike moves `max_temp_c` and the trend.
- `reading_count` depends on the sensor's reporting cadence; a sensor reporting at a different rate would shift it.
- Carrier comes from the latest reading. A mid-shipment carrier change is reflected only from then on.

Step 4 limitations:

- If audit reporting breaks its 30 h promise, the late labels are unknown at the cutoff, so those rows are labeled negative. The `beyond_reporting_lag` count makes this visible but cannot correct it.
- One malformed record or one conflicting label stops the whole build. This is deliberate, but it needs a person to fix the export.
- Event conflicts keep the first *delivered* version, so the result depends on delivery order. This is the only order-dependent behavior, and it matches the live engine.
- The reporting lag is a code constant checked against the manifest. Changing it is a deliberate one-line edit in `_labels.py`.

Step 5 limitations:

- The test set has only 20 positives, so every metric is noisy. No confidence intervals are computed (left out to keep the training step small).
- The split fraction (30 %), L2 strength (1.0), 8 °C rule threshold and 1-hour freshness cutoff are fixed choices, not tuned. There is no validation set for tuning, and none is needed for a linear model with seven features.
- A single temporal split, not rolling-origin evaluation.
- The perfect AP is a property of the synthetic data and must not be read as expected production performance.
- Tree models were not run (see "Model choice").

### Time spent

| Activity | Time |
|---|---|
| Understanding the problem and the data (brief, generator, data profiling) | 2 h |
| Steps 1–2: input validation, timestamps, event store | 1 h |
| Steps 3–4: shared features, training rows, labels, censoring | 1 h |
| Step 5: training, evaluation, artifact | 1 h |
| Steps 6–9: engine prototype, demo, results (engine descoped at the timebox) | 1.5 h |
| Step 10: final documentation | 0.5 h |
| **Total** | **7 h** |

- **How the work was done:** code and tests were drafted by Claude to speed up the process. I validated them, and the changes I asked for were implemented by Claude: a simpler training step (no tree-model comparison; at the end also no bootstrap intervals, no separate metrics file and a shorter row check), the engine reduced to a prototype when the timebox was nearly reached, and finally fitting, preprocessing and metrics switched to scikit-learn, with the model saved as a pickled pipeline plus JSON metadata. Details are in TOOL_USE.md.
- **Reproducibility check:** the pure-Python version reproduced the same `model_version` (`logreg-bec4ef1481f681cf`) in the assistant's sandbox (Python 3.12.3) and on my machine (Python 3.12.8). The final pickled-pipeline version gives `logreg-7170fa340d6fa4bd`, confirmed on my machine (Python 3.12.8) with the pinned dependencies, identical to the sandbox.

Engine prototype limitations:

- **Snapshots** have no checksum, and restore only lightly validates the file structure (each record is still fully re-validated on re-ingest).
- **There is no "late data" flag:** a score for a moment later than all data received so far is not marked provisional.
- **The evicted-ID list is bounded:** a very late record for a shipment evicted long ago is accepted as a new shipment (shown in the demo).
- **A full shipment** (1,000 revisions) rejects every further record, even exact duplicates, and its scores are not flagged.
- **Stats** are flat counters, without a breakdown by error code or reason.
- **Tests:** concurrency is covered only by a 3-thread test; there is no automated harness for the generator variants (they were run by hand at step 4).

### Done vs remaining work

| Area | Status |
|---|---|
| Input validation, UTC timestamps, strict JSON | **Done**, tested |
| Event store: as-of revisions, duplicates, conflicts, device-clock policy | **Done**, tested (incl. a brute-force oracle) |
| Shared features + digest (offline = online by construction) | **Done**, tested |
| `build_training_rows`: labels, maturity, censoring, counts report | **Done**, tested |
| `train`: purged temporal split, scikit-learn logistic-regression pipeline, 2 baselines, AP/Brier, 2 slices, versioned artifact with integrity check | **Done**, tested |
| Engine `ingest`/`score`/`stats` with offline/online parity | **Working prototype**; parity proven on all 720 decisions |
| Snapshot/restore, entity bound, history cap, safe reload | **Working prototype**; replay comparison identical |
| One-command demo, RESULTS, CUSTOMER_NOTE | **Done** |

**Remaining, in priority order:**

1. Snapshot checksum and full structural validation on restore.
2. A late-data watermark: flag scores for moments beyond the data received so far.
3. Durable eviction memory (a larger ring or an archive lookup), so long-evicted shipments are always rejected.
4. When a shipment is full: still recognize duplicates, and flag its scores.
5. Stats broken down by error code and reason, wired to alerts.
6. Concurrency stress tests, and an automated harness for the generator variants.
8. Confidence intervals for the metrics (e.g. a shipment bootstrap), once there are enough incidents for them to matter.
9. Sign model artifacts (or go back to a JSON-only artifact), so a pickle from an untrusted source can never be loaded.
7. A test run on Python 3.11, the declared minimum (runs so far used Python 3.12.3 and 3.12.8).

## Exact reproduction commands

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
pytest                                                     # or: PYTHONPATH=src python -m unittest discover -s tests
python tools/demo.py                                       # end-to-end demo in a fresh process; writes demo_output/
python tools/shuffled_label_check.py                       # optional: shuffled-label sanity check (reads data/ only)
```

Optional check that the sample is unmodified (never regenerate over `data/`):

```bash
python tools/generate_dataset.py --seed 1729 --size 240 --output regen_check
python -c "from pathlib import Path; n=['events.jsonl','labels.jsonl','decisions.jsonl','MANIFEST.json']; print(all(Path('data',f).read_text().splitlines()==Path('regen_check',f).read_text().splitlines() for f in n))"
```
