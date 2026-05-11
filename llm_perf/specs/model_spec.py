
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
class MLASpec:
    """Multi-head Latent Attention configuration (DSv3-style).

    When present on `LlmModelSpec.mla`, replaces the GQA per-head K/V
    accounting with a head-shared latent representation. The per-token
    KV cache stores only the joint latent `c_KV` of dimension
    `d_c + d_qk_rope`, not per-head K and V. Per-layer attention
    parameters are the sum of six matrices (W_DQ, W_UQ, W_DKV, W_UK,
    W_UV, W_O); see `documentation/modeling/attention.md §3.3`.

    When `mla` is set, the GQA-derived `n_kv` field on `LlmModelSpec`
    is ignored for KV-cache and attention-parameter accounting.
    """

    d_c: int           # KV latent dimension (head-shared)
    d_q_c: int         # Query latent dimension (down-projection of Q)
    d_qk_nope: int     # Non-positional Q / K per-head dimension
    d_qk_rope: int     # RoPE-positional Q / K per-head dim (head-shared on K side)
    d_v: int           # Value per-head dimension

    def kv_bytes_per_token_per_layer(self, bytes_per_param: float) -> float:
        """Per-token-per-layer KV cache base for MLA: `(d_c + d_qk_rope) * b`.

        See `attention.md §3.4`. Note no factor of 2: MLA caches a single
        joint latent, not separate K and V.
        """
        return (self.d_c + self.d_qk_rope) * bytes_per_param

    def per_layer_attn_params(self, H: int, n_q: int) -> int:
        """Per-layer attention parameter count: sum of six matrices.

        See `attention.md §3.3` for the derivation.
        """
        return (
            self.per_layer_attn_params_replicated(H, n_q)
            + self.per_layer_attn_params_shardable(H, n_q)
        )

    def per_layer_attn_params_replicated(self, H: int, n_q: int) -> int:
        """Down-projection params: W_DQ + W_DKV.

        Not head-structured — under TP-attention these are replicated on
        every rank (`attention.md §3.6`). `n_q` is unused but retained for
        API symmetry with `per_layer_attn_params_shardable`.
        """
        del n_q  # unused — down-projections do not depend on head count
        return (
            H * self.d_q_c                       # W_DQ
            + H * (self.d_c + self.d_qk_rope)    # W_DKV
        )

    def per_layer_attn_params_shardable(self, H: int, n_q: int) -> int:
        """Up-projection + output params: W_UQ + W_UK + W_UV + W_O.

        Head-structured — under TP-attention these divide by G_TP across
        ranks (`attention.md §3.6`). Independent of execution mode (the
        materialized vs absorbed distinction in §3.5 affects only when
        W_UK / W_UV multiply by data, not whether they exist).
        """
        d_qk = self.d_qk_nope + self.d_qk_rope
        return (
            self.d_q_c * n_q * d_qk              # W_UQ
            + n_q * self.d_c * self.d_qk_nope    # W_UK
            + n_q * self.d_c * self.d_v          # W_UV
            + n_q * self.d_v * H                 # W_O
        )


@dataclass
class LlmModelSpec:
    """Transformer / LLM architecture spec."""

    name: str

    # Core transformer sizes
    L: int                   # number of transformer layers
    H: int                   # hidden size
    n_q: int                 # query heads
    n_kv: int                # KV heads (for GQA); ignored when `mla` is set
    I_dense: int             # FFN dim for dense layers
    vocab_size: int          # V

    # Context & numerical precision
    max_seq_len: int         # maximum sequence length (S_max)
    bytes_per_param: float   # bytes per parameter (e.g. 2 for bf16)

    # Optional MoE configuration
    moe: Optional[MoESpec] = None

    # Optional MLA configuration (DSv3-style multi-head latent attention)
    mla: Optional[MLASpec] = None

    def d_head(self) -> float:
        """Head dimension d_head = H / n_q."""
        return self.H / self.n_q

    def H_kv(self) -> float:
        """KV projection dimension H_kv = n_kv * d_head (GQA / MHA path)."""
        return self.n_kv * self.d_head()
