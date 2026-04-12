# Curiosity — Hardware Specification

## Sole Physical Entity: DGX GB10 (Spark)

All of Curiosity runs on a single DGX GB10. No external dependencies.

**Specs:**
- GPU: NVIDIA GB10 (Blackwell)
- Unified Memory: 121GB
- SSH: `rawdata@100.78.161.126`
- User: `rawdata`

## Memory Budget

```
Qwen3-72B (Server)    ~70GB   (fp4 quantized via vLLM)
ChromaDB              ~2GB    
Redis                 ~1GB    
All daemons           ~2GB    
Fine-tune workspace   ~46GB   (available for QLoRA runs triggered by Solver)
```

## Process Map

```
Port 8001   → vLLM Server (inference endpoint)
Port 6379   → Redis (message bus)
Port 8000   → ChromaDB (vector memory)

/data/curiosity/          → persistent data root
/home/rawdata/curiosity/  → daemon code
```

## Service Management

All daemons run as systemd user services. The loop survives host reboots.

```bash
# Start all
systemctl --user start curiosity.target

# Individual daemons
systemctl --user start curiosity-assessor
systemctl --user start curiosity-formulator
systemctl --user start curiosity-solver
systemctl --user start curiosity-verifier
systemctl --user start curiosity-memorizer
systemctl --user start curiosity-meta
```

## Backup

Checkpoints stored locally at `/data/curiosity/checkpoints/`.
Git repo at github.com/r4wf0d0g23/Curiosity is the code source of truth.
Memory store backed up daily to `/data/curiosity/backups/`.
