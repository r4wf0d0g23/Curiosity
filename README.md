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

## Quick Start (DGX GB10)

```bash
# 1. Clone
git clone https://github.com/r4wf0d0g23/Curiosity ~/curiosity_code
cd ~/curiosity_code

# 2. Install deps
bash scripts/install_deps.sh

# 3. Start infrastructure (Redis + ChromaDB must already be running)
# See docs/hardware.md

# 4. Start Curiosity
bash scripts/start_curiosity.sh

# 5. Check status
bash scripts/status_curiosity.sh
```

## Status

`v0.0.1` — Architecture phase. Scaffolding in progress.
