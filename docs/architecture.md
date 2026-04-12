# Curiosity — System Architecture

**Version:** 0.0.1  
**Date:** 2026-04-11  
**Hardware:** DGX GB10 (sole physical entity)  
**Repo:** https://github.com/r4wf0d0g23/Curiosity

---

## Core Principle

The system is separated into two layers:

1. **The Server** — the target model. The thing being improved.
2. **Curiosity** — the improver. Never modifies itself, only the Server.

This separation is the source of stability. The system that evaluates is never the system being evaluated.

---

## The Two Layers

### Layer 0: The Server

Single inference endpoint. Hosts the target model (initial: Qwen3-72B on DGX GB10 via vLLM).

Responsibilities:
- Serve inference requests from Curiosity daemons
- Accept verified weight modifications from the Verifier
- Maintain a versioned checkpoint store
- Never modify itself

```
THE SERVER
├── inference_endpoint       → vLLM HTTP API (port 8001)
├── modification_endpoint    → accepts verified patches from Verifier
├── snapshot_store           → /data/curiosity/checkpoints/
└── current_state            → active model weights
```

### Layer 1: Curiosity (Six Daemons + Meta)

Six independent processes communicating via a persistent message bus. No daemon is required for the others to continue running — failure is isolated, not cascading.

---

## The Six Daemons

### 1. ASSESSOR
*The curious one.*

Runs continuously. Two parallel modes:

**Gap Mode** — probes the Server against a growing benchmark suite. Identifies where the current model fails. Measures regression after every modification.

**Curiosity Mode** — generates probes into unexplored territory. Asks: *what haven't I tested yet?* Rewards novelty — a domain the model has never been evaluated on is intrinsically interesting regardless of pass/fail outcome. This is what prevents the system from converging on local optima.

Output: ranked signal stream (gap signals + curiosity signals) → `ASSESS_QUEUE`

### 2. FORMULATOR
*The one that defines the problem.*

Converts Assessor signals into **Problem Packets** with three mandatory fields:

1. **Problem** — precise capability description
2. **Success Criterion** — automatable test. If this cannot be defined, the problem is deprioritized.
3. **Scope** — swim (targeted one-time fix) or bridge (generalizable recurring solution)

Priority = severity × frequency × novelty. High-severity, recurring, novel problems first.

Output: prioritized Problem Packets → `FORMULATE_QUEUE`

### 3. SOLVER
*The one that finds the path.*

First action: query Memory. Has this problem class been solved before? If yes — retrieve and adapt. If no — generate novel approaches.

Produces a **Solution Plan**:
- The exact modification to apply
- The checkpoint to create first (rollback anchor)
- The expected outcome

The Solver proposes. The Verifier decides. The Solver never touches the Server directly.

Output: Solution Plans → `SOLVE_QUEUE`

### 4. VERIFIER
*The one that closes the loop.*

Receives Solution Plans. Creates checkpoint. Applies modification to Server. Runs success criterion from Problem Packet. Runs full regression suite.

Two outcomes only:
- **PASS** → commit modification, send to Memorizer
- **FAIL** → automatic rollback to checkpoint, failure data returned to Solver

The Verifier has absolute veto. Nothing reaches the Server permanently without passing here.

Output: Verified results → `MEMORIZE_QUEUE`

### 5. MEMORIZER
*The one that remembers the path.*

Stores everything: Problem Packets, Solution Plans, outcomes — pass and fail. Failure paths are as valuable as success paths; they eliminate search space.

Indexes by: problem type, domain, solution type, outcome.

Runs the **improvement loop** asynchronously: takes verified solutions and searches for better versions. Updates the stored path when a better solution is found.

Output: discoveries, improvement results → `ASSESS_QUEUE` (loop closure)

### 6. META-ARCHITECT
*The one that improves Curiosity itself.*

Watches system performance. Identifies new problem types outside current daemon coverage. Spawns new specialist daemons. Retires underperforming daemons. Adjusts Formulator priority weights based on what produces the most improvement per cycle.

Curiosity improves the Server. Meta-Architect improves Curiosity.

---

## The Message Bus

Persistent. Ordered. Survives daemon crashes and host reboots.

```
ASSESS_QUEUE      ← Memorizer, Meta-Architect feed back here (loop closure)
FORMULATE_QUEUE   ← Assessor outputs here
SOLVE_QUEUE       ← Formulator outputs here
VERIFY_QUEUE      ← Solver outputs here
MEMORIZE_QUEUE    ← Verifier outputs here
```

Technology: Redis Streams (persistent, ordered, consumer groups for daemon recovery).

---

## The Full Loop

```
                    CURIOSITY MODE
                    (novelty signal)
                         ↓
         ┌──────── ASSESSOR ←──────────────────────┐
         │         GAP MODE                         │
         │              ↓                           │
         │         FORMULATOR                       │
         │         (Problem Packet)                 │
         │              ↓                           │
         │           SOLVER ←── MEMORIZER           │
         │           (Solution Plan)    ↑           │
         │              ↓              │            │
         │          VERIFIER ─────────→             │
         │          PASS: commit                    │
         │          FAIL: rollback → Solver         │
         │              ↓                           │
         └─────── SERVER MODIFIED ─────────────────→ loop
         
         META-ARCHITECT watches entire loop
         and modifies Curiosity's own daemons
```

---

## Data Architecture

```
/data/curiosity/
├── checkpoints/            # versioned Server snapshots (ZFS snapshots)
│   ├── v0.0.1/
│   ├── v0.0.2/
│   └── ...
├── memory/
│   ├── problems/           # Problem Packets, indexed
│   ├── solutions/          # Solution Plans, indexed
│   ├── outcomes/           # pass/fail records
│   └── paths/              # problem→solution→outcome chains
├── benchmarks/             # Assessor probe library (grows over time)
├── bus/                    # Redis Streams persistent data
└── logs/                   # per-daemon structured logs
```

---

## Hardware: DGX GB10

Single physical entity. All processes run here.

```
DGX GB10 (121GB unified memory)
├── The Server        → vLLM, port 8001 (Qwen3-72B, ~70GB)
├── ASSESSOR          → lightweight Python daemon
├── FORMULATOR        → lightweight Python daemon  
├── SOLVER            → medium (queries Server for generation)
├── VERIFIER          → medium (runs benchmark suites)
├── MEMORIZER         → lightweight + vector DB (ChromaDB)
├── META-ARCHITECT    → lightweight Python daemon
├── Redis             → message bus
└── ChromaDB          → vector memory store
```

Remaining unified memory (~50GB) available for fine-tuning runs triggered by the Solver.

---

## Versioning Protocol

Every system state is versioned. See [versioning.md](versioning.md).

- `v0.x.x` — Architecture and scaffolding phase
- `v1.x.x` — First working loop (Gap Mode only, single domain)
- `v2.x.x` — Curiosity Mode active, multi-domain
- `v3.x.x` — Meta-Architect active, self-modifying loop
- `v4.x.x` — Fully autonomous, improvement compounding

---

## What Curiosity Is Not

- Not a chatbot wrapper
- Not a fine-tuning pipeline with human review
- Not a benchmark runner
- Not a single model

Curiosity is an operating system for intelligence improvement. The model is just what it runs on.
