"""
Curiosity — Shared Types
All inter-daemon data structures.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
from datetime import datetime
import uuid


@dataclass
class ProblemPacket:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # From Assessor
    domain: str = ""                    # e.g. "math", "code", "security"
    description: str = ""               # precise capability description
    failure_rate: float = 0.0           # 0.0 - 1.0
    frequency: int = 0                  # how often this problem type appears
    novelty_score: float = 0.0          # 0.0 - 1.0, distance from explored territory
    priority_score: float = 0.0         # severity × frequency × novelty
    
    # From Formulator
    success_criterion: str = ""         # automatable test definition
    criterion_type: str = ""            # "benchmark", "unit_test", "llm_judge"
    scope: Literal["swim", "bridge"] = "swim"
    source: Literal["gap", "curiosity"] = "gap"


@dataclass
class SolutionPlan:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    problem_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # From Solver
    approach: Literal["weight_edit", "lora_finetune", "prompt_patch", "composite"] = "weight_edit"
    description: str = ""
    modification_spec: dict = field(default_factory=dict)  # approach-specific params
    expected_outcome: str = ""
    checkpoint_id: str = ""             # set by Verifier before applying
    from_memory: bool = False           # was this retrieved from Memory or generated novel?
    memory_solution_id: Optional[str] = None


@dataclass 
class VerificationResult:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    solution_id: str = ""
    problem_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # From Verifier
    outcome: Literal["pass", "fail"] = "fail"
    criterion_score: float = 0.0        # 0.0 - 1.0
    regression_detected: bool = False
    regression_details: str = ""
    checkpoint_id: str = ""             # checkpoint created before modification
    rolled_back: bool = False
    failure_mode: str = ""              # if fail: what went wrong


@dataclass
class TrainingJob:
    """Represents a LoRA fine-tuning job dispatched to the Trainer daemon."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    problem_id: str = ""
    solution_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Training parameters
    training_domain: str = ""
    n_pairs: int = 100                  # number of Q&A training pairs to generate
    lora_rank: int = 16                 # LoRA rank (r)
    lora_alpha: int = 32                # LoRA alpha scaling
    epochs: int = 2                     # training epochs
    learning_rate: float = 2e-4
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])
    batch_size: int = 4

    # Runtime state
    status: Literal["queued", "generating_data", "training", "loading", "done", "failed"] = "queued"
    adapter_path: str = ""              # set after training completes
    adapter_name: str = ""              # vLLM adapter identifier
    error: str = ""
    duration_sec: float = 0.0

    # Problem context (carried from ProblemPacket for data generation)
    success_criterion: str = ""
    description: str = ""


@dataclass
class MemoryPath:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    problem: ProblemPacket = field(default_factory=ProblemPacket)
    solution: SolutionPlan = field(default_factory=SolutionPlan)
    result: VerificationResult = field(default_factory=VerificationResult)
    
    # Improvement tracking
    improved_by: Optional[str] = None   # ID of improved solution, if any
    improvement_of: Optional[str] = None # ID of prior solution this improves
