# Curiosity — Versioning Protocol

## Semantic Versioning

`MAJOR.MINOR.PATCH`

- **MAJOR** — architectural phase (see roadmap)
- **MINOR** — new daemon capability or domain added
- **PATCH** — bug fix, config change, or minor improvement

## Server Checkpoint Versioning

Every modification to the Server creates a named checkpoint before applying:

```
checkpoint_id: curiosity-{version}-{timestamp}-{problem_hash}
location: /data/curiosity/checkpoints/{checkpoint_id}/
contents:
  - model_weights/     # full weight snapshot or LoRA adapter
  - problem_packet.json
  - solution_plan.json
  - verification_result.json
```

Rollback is atomic: restore weights from checkpoint, mark solution as FAILED in Memory.

## Git Versioning

All code, configs, and architecture docs versioned at:
https://github.com/r4wf0d0g23/Curiosity

Tag every working loop milestone:
- `v1.0.0` — first successful end-to-end cycle
- `v2.0.0` — curiosity mode confirmed active
- etc.

## Memory Versioning

The Memory store is append-only. Nothing is ever deleted — failed solutions are retained as negative examples. The improvement loop creates new entries rather than overwriting old ones.
