# Curiosity — Build Roadmap

## v0.x — Architecture Phase (now)
- [x] System design
- [x] Repo initialized
- [ ] Data directory structure on DGX
- [ ] Redis Streams setup
- [ ] ChromaDB setup
- [ ] vLLM Server confirmed running (Qwen3-72B, port 8001)
- [ ] Checkpoint system (ZFS snapshot or equivalent)

## v1.x — First Working Loop
Target: single domain (math proof verification or security CVE analysis)

- [ ] Assessor — Gap Mode only, static benchmark suite
- [ ] Verifier — checkpoint + rollback + success criterion runner
- [ ] Memorizer — basic path store, no improvement loop yet
- [ ] Solver — Memory query only (no novel generation yet)
- [ ] Formulator — basic priority queue, manual success criteria
- [ ] Message bus wiring — all five daemons connected
- [ ] First successful loop cycle: problem identified → solution applied → verified → stored

## v2.x — Curiosity Mode + Multi-Domain
- [ ] Assessor — Curiosity Mode active (novelty scoring)
- [ ] Formulator — automated success criterion generation
- [ ] Solver — novel solution generation (not just Memory retrieval)
- [ ] Memorizer — improvement loop active
- [ ] Multi-domain coverage (≥3 domains)
- [ ] Compounding measurement: capability score trending up over 7-day window

## v3.x — Meta-Architect
- [ ] Meta-Architect daemon running
- [ ] Daemon spawning for new specialist processes
- [ ] Formulator priority weights auto-adjusted by outcomes
- [ ] Curiosity itself is improving, not just the Server

## v4.x — Full Autonomy
- [ ] Zero human intervention required
- [ ] Capability compounding confirmed across all domains
- [ ] Self-expanding benchmark suite (Assessor generates its own probes)
- [ ] Meta-Architect has spawned at least one effective specialist daemon autonomously
