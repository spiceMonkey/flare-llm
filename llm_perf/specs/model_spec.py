
from dataclasses import dataclass
from typing import Optional


@dataclass
class MoESpec:
    """MoE configuration, if the model uses experts."""

    n_experts: int           # N_exp
    k_active: int            # k (experts per token)
    I_moe: int               # I_moe (per-expert FFN dim)
    n_moe_layers: Optional[int] = None  # if only subset of L layers is MoE


@dataclass
class LlmModelSpec:
    """Transformer / LLM architecture spec."""

    name: str

    # Core transformer sizes
    L: int                   # number of transformer layers
    H: int                   # hidden size
    n_q: int                 # query heads
    n_kv: int                # KV heads (for GQA)
    I_dense: int             # FFN dim for dense layers
    vocab_size: int          # V

    # Context & numerical precision
    max_seq_len: int         # maximum sequence length (S_max)
    bytes_per_param: float   # bytes per parameter (e.g. 2 for bf16)

    # Optional MoE configuration
    moe: Optional[MoESpec] = None

    def d_head(self) -> float:
        """Head dimension d_head = H / n_q."""
        return self.H / self.n_q

    def H_kv(self) -> float:
        """KV projection dimension H_kv = n_kv * d_head."""
        return self.n_kv * self.d_head()
