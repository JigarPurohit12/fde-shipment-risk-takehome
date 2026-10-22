# Forward Deployed Engineer Take-Home
## Point-in-Time Shipment Risk

## Working agreement

**Role: Forward Deployed Engineer**  
**Timebox: 7 hours maximum. Python 3.11+.** Stop at the timebox and identify unfinished work. A small, correct vertical slice with explicit limitations is preferable to an incomplete platform. No UI, cloud deployment, paid services, GPU, or presentation slides are required. Everything must run locally without network access after dependency installation.

The customer notes below may conflict. Resolve them explicitly in `DECISIONS.md`; explain the customer impact of your choice. No clarification is available during the exercise. Do not hard-code IDs, dates, fixture sizes, or generated answers. Additional evaluation may use unseen data with the same schema and documented semantics. There is no minimum model accuracy target.


## Customer scenario

A cold-chain customer needs the probability of a temperature excursion in the next six hours. Sensor measurements arrive late, repeat, and receive corrections. Audit labels become available after the incident. Operations needs reproducible historical decisions and safe concurrent model reloads.

## Supplied data and semantics

`events.jsonl`: `event_id`, `revision` (positive integer), `entity_id` (shipment), `event_at`, `received_at`, `kind="temperature_c"`, numeric `value`, `carrier`. Identity is `(event_id, revision)` within an entity. Higher revisions supersede lower ones only from their receipt time onward. Same identity with different content is an invalid conflict. Delivery order is file order, which is deliberately not chronological.

`labels.jsonl`: `incident_id`, `entity_id`, `incident_at`, `label_available_at`. An incident is positive for a decision at t iff it occurs in `(t, t+6h]`. `decisions.jsonl`: `entity_id`, `as_of`. The manifest observation cutoff is the knowledge cutoff for the supplied training export. Audit reporting is guaranteed complete only for incidents at least 30 hours before that cutoff (`label_reporting_lag_hours`). For evaluation eligibility require `as_of + 6h + 30h <= observation_end` for every row, including positives. During training, apply the same maturity rule using your actual training cutoff, and require label availability by that cutoff. Records after the observation cutoff are supplied only for replay/future-stream tests.

## Required work

1. Implement `build_training_rows(events, labels, decisions, observation_end)` returning dictionaries with `entity_id`, `as_of`, `features`, `label`, and `metadata`. Explain censored decisions and preserve historical revisions: a correction received tomorrow cannot rewrite today's features. Use receipt time for availability; define how future or invalid device clocks affect features.
2. Implement `train(rows, artifact_dir)` with a lightweight model and a self-contained versioned artifact. Report a temporal split, delayed-label eligibility, average precision, Brier score, a baseline, and two meaningful slices. Account for overlap between six-hour label windows across split boundaries.
3. Implement `RiskEngine`. `ingest(record)` returns True iff accepted effective state changes. `score(entity_id, as_of)` returns the specified result dictionary with a finite probability in [0,1]. Offline and online features must agree for the same known records and time.
4. Support atomic deterministic snapshots, restore, bounded entity state, and concurrent scoring/reload. A failed reload keeps the previous artifact active. Retain sufficient revision history for supported historical queries; explicitly reject queries outside your retention policy. State what happens to a late correction after eviction.
5. Demonstrate duplicate delivery, correction arrival after a decision, a reversed revision order, an immature label window, eviction, and failed reload. Compare uninterrupted replay with snapshot/restore replay.

## Customer implementation notes

1. “Use the latest corrected value when replaying every old decision.”
2. “Randomly split decision rows; the same shipment may appear in both sets.”
3. “If model loading fails, report zero risk so customer dashboards remain green.”
4. “Keep every event forever; enforce memory only on the prediction cache.”

## Public interface

Implement `src/shipment_risk/solution.py`. Records are mappings with the JSON fields above. Input record timestamps are ISO 8601 strings with an explicit UTC offset. `as_of` method arguments and `observation_end` are aware Python `datetime` objects.

Construct the engine with `(artifact_dir: Path, max_entities: int = 10000)`. `ingest(record) -> bool`, `score(entity_id, as_of) -> Result`, `snapshot(destination: Path) -> None`, `restore(artifact_dir: Path, snapshot: Path) -> engine`, `reload_model(artifact_dir: Path) -> bool`, and `stats() -> Mapping` are public. `stats` must include `resident_entities`. The memory limit applies to resident entities and their in-memory indexes, not only outputs. Per-entity event/history retention must also be bounded and documented.

`Result` wraps `entity_id` (the invoice ID for reconciliation; query ID for search), `as_of` (UTC ISO 8601 string), and a `payload` mapping. Required payload fields for `score`: `probability: float, model_version: str, feature_digest: str, degraded: bool, reasons: list[str]`. Extra payload fields are allowed. Return canonical bytes through `Result.to_wire()`. Explain all statuses, reason codes, and error types. Input mappings must not be mutated.

## Evidence of your work and tool use

AI assistance is permitted and must be disclosed. In `TOOL_USE.md`, name tools/models used, describe which parts they assisted with, and include up to three representative prompts or interaction excerpts with a short explanation of how you checked the resulting work. Redact personal information, credentials, and unrelated content. If you did not use AI, state that. Also disclose copied code and its source/license. You are responsible for understanding and validating every submitted line.

Keep a short `WORKLOG.md` with three to six actual development checkpoints: what you tried, what failed or changed, and the evidence behind your next step. Do not reconstruct a fictional history. Commit history is optional; a private account, full chat export, screen recording, or personal browsing history is not required.

## Submission

Submit a repository or ZIP containing implementation, declared dependencies, tests, a runnable demo, and the completed decision record. Include `RESULTS.md` with exact commands and actual measured results, plus `CUSTOMER_NOTE.md` (maximum 400 words) explaining what is ready, the launch risk, and a practical rollout/rollback plan. Include your time spent and remaining limitations. Do not include secrets or a local virtual environment.

The provided interfaces are intentionally unimplemented. Public tests check only the starter contract; they do not establish correctness. Add meaningful tests of your implementation. Preserve public signatures and result fields; add private helpers or extra artifact files as needed. All public methods must reject malformed inputs explicitly rather than silently fabricating successful results. Use UTC-aware timestamps in outputs; equivalent timestamp offsets represent the same instant. Canonical JSON must sort object keys, preserve list order, reject NaN/infinity, and contain no wall-clock metadata.

Set up and inspect the sample:

```bash
python tools/generate_dataset.py --seed 1729 --size 240
python -m pip install -e '.[dev]'
pytest
# Starter checks can also run without pytest:
PYTHONPATH=src python -m unittest discover -s tests
```

The data is synthetic and intentionally imperfect. `data/MANIFEST.json` records the generator seed, size, files, and observation cutoff. JSONL records use the schema below. The generator uses only the Python standard library. Document any change to generated data; never manually repair the sample to make your implementation pass. Provide one exact command to run your end-to-end demo in a fresh process.

## Technical follow-up

In a 60-minute session, you will reproduce a result, trace a supplied record through your code, diagnose a failed invariant, and implement a small requirement change. Be prepared to explain the tests and tradeoffs, including any AI-assisted code. We assess correctness, evidence, operational judgment, and your ability to adapt the implementation.
#   f d e - s h i p m e n t - r i s k - t a k e h o m e  
 