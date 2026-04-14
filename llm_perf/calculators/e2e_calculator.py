
from dataclasses import dataclass
from typing import Optional

from ..specs.model_spec import LlmModelSpec
from ..specs.system_spec import SystemSpec
from ..specs.partition_spec import PartitionSpec
from ..specs.tuner_spec import TuningSpec
from ..specs.overhead_spec import OverheadSpec
from ..utils import GB_TO_BYTES
from .inference_calculator import InferenceResults
from .prefill_calculator import PrefillResults


US_TO_S = 1e-6


@dataclass
class E2EResults:
    # TTFT variants (seconds)
    TTFT_single: float          # t_sched + t_prefill + t_token (co-located)
    TTFT_disagg: float          # + t_KV_transfer
    TTFT_chunked: float         # t_sched + Σ t_chunk^(k) + t_token

    # TPOT (seconds)
    TPOT: float                 # t_token(B) / B

    # Throughput
    throughput_per_gpu: float   # tokens/s per GPU
    interactivity: float        # 1 / TPOT (tokens/s perceived by user)
    TTPS: float                 # total cluster throughput

    # KV transfer (disaggregated)
    t_KV_transfer: float        # seconds (0 if co-located)
    M_KV_transfer: float        # bytes

    # Framework overhead breakdown
    t_sched: float              # seconds
    t_framework_per_step: float # per-step overhead (seconds)

    def e2e_latency(self, N_out: int) -> float:
        """E2E latency = TTFT + (N_out - 1) * TPOT."""
        return self.TTFT_single + (N_out - 1) * self.TPOT


class E2ECalculator:
    """Assemble end-to-end metrics from decode + prefill + overhead (documentation/modeling/e2e.md)."""

    def __init__(
        self,
        decode_results: InferenceResults,
        prefill_results: Optional[PrefillResults],
        overhead: OverheadSpec,
        model: LlmModelSpec,
        system: SystemSpec,
        partition: PartitionSpec,
        tuner: TuningSpec,
    ) -> None:
        self.decode = decode_results
        self.prefill = prefill_results
        self.overhead = overhead
        self.model = model
        self.system = system
        self.partition = partition
        self.tuner = tuner

    def run(self) -> E2EResults:
        oh = self.overhead
        dec = self.decode.latency

        # Framework overhead
        t_sched = (oh.t_sched_us + oh.t_tok_us) * US_TO_S
        t_framework_per_step = (oh.t_graph_us + oh.t_sample_us + oh.t_detok_us) * US_TO_S

        # Prefill latency (0 if no prefill results)
        if self.prefill is not None:
            t_prefill = self.prefill.latency.t_prefill
            t_prefill_chunked = self.prefill.latency.t_prefill_chunked
        else:
            t_prefill = 0.0
            t_prefill_chunked = 0.0

        # First decode step latency
        t_first_token = dec.t_token

        # KV transfer (disaggregated)
        M_KV_transfer = 0.0
        t_KV_transfer = 0.0
        if oh.disagg_bandwidth_GBps > 0 and self.prefill is not None:
            S = self.tuner.S_input
            H_kv = self.model.H_kv()
            L = self.model.L
            b = self.model.bytes_per_param
            PP = self.partition.PP
            TP = self.partition.TP
            SP = self.partition.SP
            # Total KV bytes produced by prefill
            M_KV_transfer = 2 * L * S * H_kv * b
            t_KV_transfer = (
                oh.disagg_alpha_us * US_TO_S
                + M_KV_transfer / (oh.disagg_bandwidth_GBps * GB_TO_BYTES)
            )

        # TTFT assembly (documentation/modeling/e2e.md §2)
        TTFT_single = t_sched + t_prefill + t_first_token
        TTFT_disagg = TTFT_single + t_KV_transfer
        TTFT_chunked = t_sched + t_prefill_chunked + t_first_token

        # TPOT (documentation/modeling/e2e.md §3)
        TPOT = dec.TPOT

        # Throughput (documentation/modeling/e2e.md §5)
        TTPS = dec.TTPS
        throughput_per_gpu = TTPS / self.system.num_devices if self.system.num_devices > 0 else 0.0
        interactivity = 1.0 / TPOT if TPOT > 0 else 0.0

        return E2EResults(
            TTFT_single=TTFT_single,
            TTFT_disagg=TTFT_disagg,
            TTFT_chunked=TTFT_chunked,
            TPOT=TPOT,
            throughput_per_gpu=throughput_per_gpu,
            interactivity=interactivity,
            TTPS=TTPS,
            t_KV_transfer=t_KV_transfer,
            M_KV_transfer=M_KV_transfer,
            t_sched=t_sched,
            t_framework_per_step=t_framework_per_step,
        )
