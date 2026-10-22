# Customer note: point-in-time shipment risk

**What is ready**

- A score per shipment: the probability of a temperature excursion in the next six hours, using only sensor data the platform had actually received at that moment. Late, repeated and corrected readings are handled, and a correction never rewrites an earlier decision, so historical scores are reproducible for audits.
- Training and live scoring use the same code: all 720 historical decisions score identically in both. The model is tested on later shipments than it learned from, with a 36-hour gap so outcomes cannot leak across.
- On the sample, the model ranks every excursion above every non-excursion. That reflects how clean the synthetic data is; expect lower numbers on real data.

**Where we deliberately differ from your notes**

- Old decisions are replayed with the data known at the time, not later corrections, so audits stay truthful.
- Evaluation is by time, not random rows, so the numbers are not over-optimistic.
- A model failure never shows "zero risk": the engine refuses to start, or keeps the previous model and records the failure for alerting.
- Memory is bounded: the engine keeps a limited number of active shipments and refuses older ones rather than answer from partial data; they can be replayed offline from the event export.

**Launch risk**

- The live engine is a working prototype. Still missing: integrity checks on saved snapshots, a "data may still be arriving" flag, and long-term memory of evicted shipments.
- The test set has only 20 excursions, so every number carries real uncertainty.
- A score marked "degraded" means "no reliable answer" and must never be displayed as green.

**Rollout and rollback**

1. Shadow mode for at least two weeks: scores are logged next to your existing 8 °C alarms, and no actions are taken from them.
2. Advisory: the probability is shown beside the alarms, which stay on.
3. Alerting, one carrier first, once the missing items are done and the shadow results beat the alarm rule.

Rollback at any stage: switch back to the previous model version without a restart, or hide the scores. Your existing alarms are never switched off.

**Time spent:** 7 hours (2 understanding the problem and data, 3 on the data and model pipeline, 1.5 on the engine prototype and demo, 0.5 on documents). Full limitations and remaining work are in DECISIONS.md.
