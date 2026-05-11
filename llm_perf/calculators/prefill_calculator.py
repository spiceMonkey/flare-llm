
from dataclasses import dataclass
from typing import Optional

from ..specs.framework_spec import FrameworkSpec
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
    """Prefill performance calculator (documentation/modeling/prefill.md).

    Five-spec composition (`model × system × partition × tuner × framework`),
    matching `InferenceCalculator`. `framework` is optional; when omitted it
    defaults to `FrameworkSpec.default()` (neutral roofline).
    """

    def __init__(
        self,
        model: LlmModelSpec,
        system: SystemSpec,
        partition: PartitionSpec,
        tuner: TuningSpec,
        framework: Optional[FrameworkSpec] = None,
    ) -> None:
        self.model = model
        self.system = system
        self.partition = partition
        self.tuner = tuner
        self.framework = framework if framework is not None else FrameworkSpec.default()

    def run(self) -> PrefillResults:
        flops = compute_prefill_flops(self.model, self.partition, self.tuner, self.framework)
        traffic = compute_prefill_traffic(self.model, self.partition, self.tuner, self.framework)
        comm = compute_prefill_comm(self.model, self.system, self.partition, self.tuner, self.framework)
        latency = compute_prefill_latency(
            self.system, self.partition, self.tuner, self.model,
            flops, traffic, comm, self.framework,
        )
        return PrefillResults(
            flops=flops,
            traffic=traffic,
            comm=comm,
            latency=latency,
        )
