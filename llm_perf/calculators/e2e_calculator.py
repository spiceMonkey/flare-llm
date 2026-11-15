
from dataclasses import dataclass
from typing import Optional

from ..specs.model_spec import LlmModelSpec
from ..specs.system_spec import SystemSpec
from ..specs.partition_spec import PartitionSpec
from ..specs.tuner_spec import TuningSpec
from ..specs.overhead_spec import OverheadSpec
from ..specs.disagg_spec import DisaggSpec
from ..utils import GB_TO_BYTES
from .inference_calculator import InferenceResults
from .prefill_calculator import PrefillResults


US_TO_S = 1e-6


@dataclass
class E2EResults:
    # TTFT (seconds) — unified across co-lo and disagg branches
    TTFT: float                 # t_sched + t_prefill (+ t_handoff; + one decode step when a disaggregated decode worker emits token 1)
    TTFT_chunked: float         # TTFT with chunked prefill

    # TPOT (seconds)
    TPOT: float                 # user-observed inter-token time: t_step_user(B), or the
                                # speculative-decoding value t_step_user_verify / N_tok_per_step

    # Throughput
    throughput_per_gpu: float   # tokens/s per GPU
    interactivity: float        # 1 / TPOT (tokens/s perceived by user)
    TTPS: float                 # total cluster throughput

    # KV handoff diagnostics
    t_handoff: float            # seconds (co-lo or disagg, depending on DisaggSpec.disaggregated)
    t_repack: float             # seconds (scale-up repack component; 0 if no repack)
    M_KV_total: float           # bytes (cluster-aggregate KV cache)

    # Framework overhead breakdown
    t_sched: float              # seconds
    t_framework_per_step: float # per-step overhead (seconds)


class E2ECalculator:
    """Assemble end-to-end metrics from decode + prefill + overhead + disagg (prefill.md §6, e2e.md)."""

    def __init__(
        self,
        decode_results: InferenceResults,
        prefill_results: Optional[PrefillResults],
        overhead: OverheadSpec,
        disagg: DisaggSpec,
        model: LlmModelSpec,
        system: SystemSpec,
        partition: PartitionSpec,
        tuner: TuningSpec,
    ) -> None:
        self.decode = decode_results
        self.prefill = prefill_results
        self.overhead = overhead
        self.disagg = disagg
        self.model = model
        self.system = system
        self.partition = partition
        self.tuner = tuner

    def run(self) -> E2EResults:
        oh = self.overhead
        dg = self.disagg
        dec = self.decode.latency

        # Framework overhead.
        # `oh.t_graph_us` is the legacy flat per-step CUDA-graph constant;
        # it is superseded when the decode latency model exposes a derived
        # per-round SW budget (`LatencyResults.t_kernel > 0`, which is the
        # production default since the kernel-launch refactor). Including
        # both would double-count, so we only fold `t_graph_us` in when the
        # derived term is zero (legacy path / SW disabled).
        t_sched = (oh.t_sched_us + oh.t_tok_us) * US_TO_S
        t_graph_legacy_us = oh.t_graph_us if dec.t_kernel <= 0.0 else 0.0
        t_framework_per_step = (t_graph_legacy_us + oh.t_detok_us) * US_TO_S

        # Prefill latency (0 if no prefill results)
        if self.prefill is not None:
            t_prefill = self.prefill.latency.t_prefill
            t_prefill_chunked = self.prefill.latency.t_prefill_chunked
        else:
            t_prefill = 0.0
            t_prefill_chunked = 0.0

        # KV cache volume (cluster-aggregate, prefill.md §6.2). Latent-attention
        # models store one shared latent per token per layer, not per-head K/V.
        S = self.tuner.S_input
        b = self.model.kv_bytes()
        if self.model.mla is not None:
            M_KV_total = self.model.L * S * self.model.mla.kv_bytes_per_token_per_layer(b)
        else:
            M_KV_total = 2 * self.model.L * S * self.model.H_kv() * b

        # KV handoff: branch on DisaggSpec.disaggregated (prefill.md §6.3 / §6.4)
        t_handoff, t_repack = self._compute_handoff(M_KV_total, t_prefill)

        # TTFT assembly (prefill.md §6.5; e2e.md §2). t_prefill ends with the
        # LM head on the last prompt position, so the prefill pass itself
        # yields token 1. What follows depends on who streams it:
        #   - co-located: the same engine emits it; the repack into the
        #     decode layout (t_handoff, usually zero) is the only extra.
        #   - disaggregated, first_token="decode": the decode worker waits
        #     for the KV cache, re-runs the last prompt position (one decode
        #     step) and emits the token.
        #   - disaggregated, first_token="prefill": the prefill worker emits
        #     it at once; the transfer delays the second token instead and
        #     is reported in t_handoff only.
        if not dg.disaggregated:
            t_after_prefill = t_handoff
        elif dg.first_token == "prefill":
            t_after_prefill = 0.0
        elif dg.first_token == "decode":
            t_after_prefill = t_handoff + dec.t_step_user
        else:
            raise ValueError(
                f"DisaggSpec.first_token must be 'decode' or 'prefill', got {dg.first_token!r}"
            )
        TTFT = t_sched + t_prefill + t_after_prefill
        TTFT_chunked = t_sched + t_prefill_chunked + t_after_prefill

        # TPOT (e2e.md §3). With speculative decoding enabled the decode model
        # reports the per-accepted-token time separately; use it, and scale
        # throughput by the same ratio (N_tok_per_step tokens per verify step).
        # The draft model's own cost is not modelled, so this is an upper
        # bound on the speculative gain.
        speculative = dec.N_tok_per_step > 1.0 and dec.TPOT_spec > 0.0
        TPOT = dec.TPOT_spec if speculative else dec.TPOT

        # Throughput (e2e.md §5)
        TTPS = dec.TTPS * (dec.TPOT / dec.TPOT_spec) if speculative else dec.TTPS
        throughput_per_gpu = TTPS / self.system.num_devices if self.system.num_devices > 0 else 0.0
        interactivity = 1.0 / TPOT if TPOT > 0 else 0.0

        return E2EResults(
            TTFT=TTFT,
            TTFT_chunked=TTFT_chunked,
            TPOT=TPOT,
            throughput_per_gpu=throughput_per_gpu,
            interactivity=interactivity,
            TTPS=TTPS,
            t_handoff=t_handoff,
            t_repack=t_repack,
            M_KV_total=M_KV_total,
            t_sched=t_sched,
            t_framework_per_step=t_framework_per_step,
        )

    def _compute_handoff(self, M_KV_total: float, t_prefill: float) -> tuple[float, float]:
        """Compute t_handoff and t_repack from DisaggSpec (prefill.md §6.3/§6.4)."""
        dg = self.disagg

        if dg.disaggregated:
            # §6.4 disaggregated transfer
            if dg.inter_bandwidth_GBps <= 0:
                return 0.0, 0.0
            BW_inter = dg.inter_bandwidth_GBps * GB_TO_BYTES
            alpha_eff = (dg.inter_alpha_us + dg.N_WR * dg.tau_WR_us) * US_TO_S
            t_repack = (
                M_KV_total / (dg.repack_GBps * GB_TO_BYTES) * dg.repack_eta
                if dg.repack_GBps > 0 else 0.0
            )
            t_handoff = max(
                0.0,
                alpha_eff + M_KV_total / BW_inter + t_repack - dg.overlap_rho_KV * t_prefill,
            )
            return t_handoff, t_repack

        # §6.3 co-located handoff
        t_repack = (
            M_KV_total / (dg.colo_repack_GBps * GB_TO_BYTES) * dg.colo_repack_eta
            if dg.colo_repack_GBps > 0 else 0.0
        )
        t_handoff = dg.colo_alpha_us * US_TO_S + t_repack
        return t_handoff, t_repack
