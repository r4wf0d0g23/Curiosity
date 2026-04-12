# Curiosity

> An autonomous, continuously self-improving AI system. No human gates. No time limits. No pausing to ask for directions.

Curiosity is a multi-layer daemon architecture that runs on a single physical host — the DGX GB10 — and recursively improves a target model through a closed curiosity loop:

**ASSESS → FORMULATE → SOLVE → VERIFY → MEMORIZE → ASSESS**

The loop never terminates. Curiosity is the fuel.

---

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│              THE SERVER                      │
│   Single inference endpoint. The target.     │
│   Versioned. Checkpointed. Replaceable.      │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│               CURIOSITY                      │
│   Six daemons. One message bus.              │
│   Assess → Formulate → Solve →               │
│   Verify → Memorize → Assess                 │
│                                              │
│   + Meta-Architect: improves the loop        │
└─────────────────────────────────────────────┘
```

## Docs

- [Architecture](docs/architecture.md) — full system design
- [Daemon Specs](docs/daemons.md) — per-process specifications
- [Hardware](docs/hardware.md) — DGX GB10 deployment
- [Versioning](docs/versioning.md) — checkpoint and rollback protocol
- [Roadmap](docs/roadmap.md) — build order

## Versioning

All versions tracked here: [github.com/r4wf0d0g23/Curiosity](https://github.com/r4wf0d0g23/Curiosity)

## Status

`v0.0.1` — Architecture phase. Scaffolding in progress.
