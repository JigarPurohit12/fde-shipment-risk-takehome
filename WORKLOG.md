# Development checkpoints

_Six checkpoints (the README allows three to six). Code and tests were drafted by Claude to speed up the process; I validated them, and the changes I asked for were implemented by Claude._

## 1. Brief and data profiling

- **Tried:** compared CANDIDATE_BRIEF.docx with README.md word by word (same text). Regenerated the sample with `--seed 1729 --size 240`: all four files byte-identical. Profiled the data.
- **Found:**
  - 4,402 events = 3,840 readings + 480 exact duplicates + 35 corrections + 47 backdated records.
  - 720 decisions, of which 714 are eligible and 6 censored.
  - 462 readings arrive exactly at a decision time; 175 deliveries are out of receipt order.
  - The sample has no conflicts, future device clocks, truly reversed revisions or post-cutoff records.
- **Evidence for next step:** filtering by `event_at` lets every decision of a bad shipment see a 19.0 °C reading received after the incident, so availability must use `received_at`. Defect types missing from the sample must be tested with hand-built records.
- **Who did what:** Claude explained the brief and profiled the data; I studied the brief, the data files and the generator to understand the traps.

## 2. Input validation and event store

- **Tried:** probed `datetime.fromisoformat` before choosing a parser; built a per-shipment store keeping every revision, where "known at T" means `received_at ≤ T` and the highest revision wins.
- **Changed:**
  - `fromisoformat` (Python 3.12) returned a naive datetime for `2026-01-01`, accepted week dates and silently truncated a 7-digit fraction, and `\d` matched Arabic-Indic digits. I switched to an explicit RFC 3339 regex with `[0-9]`, converting to UTC on entry.
  - For `ingest`'s True/False I derived the rule "changed iff no higher revision had already been received by this one's receipt time" and checked it with a brute-force oracle instead of trusting it. I first limited the oracle to 40 shipments; it ran in 0.76 s on all 4,402 events, so I removed the limit.
- **Evidence:** the oracle matched on every sample add and on all 120 delivery orders of a tricky hand-built set. Mutation checks: 11 rules broken one at a time, all caught.
- **Who did what:** code and tests drafted by Claude; I validated them by running the test suite and reviewing the code.

## 3. Features and training rows

- **Tried:** one feature function shared by training and scoring (6-hour look-back, digest over the features plus a schema version); label rules in their own module; one row per unique decision, built from the same `known_at` + `compute_features` path.
- **Decided:**
  - Dropped a "hours since shipment start" feature: only one decision offset is positive in the sample, so it would learn the generator's schedule.
  - Missing values stay `null`, never invented.
  - Censored rows are returned with a reason, not dropped.
  - Event conflicts keep the first version (as the engine does); label conflicts stop the build.
- **Evidence:**
  - Features are identical to a store fed only records received by `as_of` (720/720).
  - Availability by `event_at` would leak the 19.0 °C reading into 94 decisions; receipt time gives 0.
  - Labels match an independent brute-force check.
  - Six generator variants, run into scratch folders, behaved as specified (lag 48 stopped the build).
- **Who did what:** code and tests drafted by Claude; I validated them by running the test suite and reviewing the code.

## 4. Training, evaluation and artifact

- **Tried:** purged, shipment-grouped temporal split; logistic regression (first fitted with Newton's method in plain Python, later as a scikit-learn pipeline; see checkpoint 6); prevalence and 8 °C-rule baselines; a checksummed JSON artifact.
- **Investigated:** test AP was 1.0, a leakage warning. Training on shuffled labels gave AP 0.051 (below the 0.093 positive rate), and a weaker-signal variant gave AP 0.867. Conclusion: a strong synthetic signal, not a leak.
- **Failed and changed:** the mutation check showed that a naive (overflowing) sigmoid passed every test. I added an extremes test, and it is now caught.
- **Evidence:** identical artifact bytes for shuffled rows; a fresh process loads the artifact with identical probabilities; `--size 10` stops cleanly with `insufficient_training_data`.
- **Who did what:** code and tests drafted by Claude; I validated them. At my request the training step was kept simple (logistic regression plus two baselines, no tree-model comparison), since accuracy is not the evaluation target.

## 5. Engine descoped to a prototype at the timebox

- **Tried:** Claude had drafted a fuller engine with snapshot checksums and deep restore validation, a bounded-tombstone policy, a late-data watermark reason and concurrency stress tests.
- **Changed:** with about 1.5 h of the timebox left, I had not reviewed that draft, so I chose a ~130-line prototype I could fully understand.
  - It reuses the offline path and keeps all four customer-note protections.
  - Everything else is listed as remaining work in DECISIONS.md.
- **Evidence:**
  - All tests pass.
  - Online equals offline for all 720 decisions (digest and probability).
  - Mutation check: six key protections broken one at a time (zero-risk startup fallback, failed reload damaging the model, no entity bound, evicted shipment re-created by a late correction, score using data received after `as_of`, torn model/version swap), and each was caught.
- **Who did what:** at my request, Claude rebuilt the engine as a smaller prototype once the timebox was nearly reached; I validated it by running the tests and the demo.

## 6. End-to-end demo and documents

- **Tried:** one command (`python tools/demo.py`) covering every README demonstration and the replay comparison.
- **Failed and changed:** the first run sent a "late correction" to the first shipment of the stream and it was **accepted**. That shipment was evicted so long ago that the bounded eviction list had forgotten it, so the correction created a new shipment.
  - I kept that line in the demo, labelled as a known limitation, and listed it as remaining work.
  - The demo now also shows a recently evicted shipment being rejected.
- **Evidence:** demo output in RESULTS.md. The replay comparison (uninterrupted vs snapshot/restore) is identical at `max_entities` 10000 and 20.
- **Simplified at the end:** to keep the training step small enough to explain, I removed the bootstrap confidence intervals, the separate `metrics.json` file and most of the field-by-field row validation, and kept both baselines.
  - The model, the artifact and `model_version` are unchanged, and the demo output is byte-identical.
  - The split protections (purge gap, shipment grouping, labels re-checked at the split time) and NaN rejection are still each caught when removed.
- **Switched to scikit-learn at the end (my request), in two moves:**
  1. Fitting with `LogisticRegression(C=1.0)` and the metrics with `average_precision_score` and `brier_score_loss`. Same objective (a gradient-is-zero test); weights within 0.0000003 of the Newton fit.
  2. Then a `Pipeline` (`SimpleImputer` → `StandardScaler`, `OneHotEncoder`, `LogisticRegression`) saved with pickle plus a JSON metadata file.
- **Failed and changed:** my first idea was to hash the pickle file for `model_version`, but a test showed that a single load/save round trip changes the pickle's bytes. So the version is now a hash of the fitted parameters, and the pickle's SHA-256 is checked before unpickling as an integrity check.
- **Evidence:** all metrics, slices and sanity checks are identical to 6 decimals across all three versions; 179 tests pass; scoring is now about 1.5 ms per call (was about 0.05 ms). `model_version` is now `logreg-7170fa340d6fa4bd`.
- **Who did what:** demo and documents drafted by Claude. I ran the demo on my machine (Python 3.12.8) and reproduced the pure-Python `model_version` `logreg-bec4ef1481f681cf`; I later confirmed the final version `logreg-7170fa340d6fa4bd` the same way.
