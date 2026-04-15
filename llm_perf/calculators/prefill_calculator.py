
from dataclasses import dataclass
from ..specs.model_spec import LlmModelSpec
from ..specs.system_spec import SystemSpec
from ..specs.partition_spec import PartitionSpec
from ..specs.tuner_spec import TuningSpec
from ..core.prefill_model import (
    compute_prefill_flops,
    compute_prefill_traffic,
    compute_prefill_comm,
    compute_prefill_latency,
    PrefillFlopsResults,
    PrefillTrafficResults,
    PrefillCommResults,
    PrefillLatencyResults,
)


@dataclass
class PrefillResults:
    flops: PrefillFlopsResults
    traffic: PrefillTrafficResults
    comm: PrefillCommResults
    latency: PrefillLatencyResults


class PrefillCalculator:
    """Prefill performance calculator (documentation/modeling/prefill.md)."""

    def __init__(
        self,
        model: LlmModelSpec,
        system: SystemSpec,
        partition: PartitionSpec,
        tuner: TuningSpec,
    ) -> None:
        self.model = model
        self.system = system
        self.partition = partition
        self.tuner = tuner

    def run(self) -> PrefillResults:
        flops = compute_prefill_flops(self.model, self.partition, self.tuner)
        traffic = compute_prefill_traffic(self.model, self.partition, self.tuner)
        comm = compute_prefill_comm(self.model, self.system, self.partition, self.tuner)
        latency = compute_prefill_latency(
            self.system, self.partition, self.tuner, self.model,
            flops, traffic, comm,
        )
        return PrefillResults(
            flops=flops,
            traffic=traffic,
            comm=comm,
            latency=latency,
        )
