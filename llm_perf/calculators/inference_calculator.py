
from dataclasses import dataclass
from typing import Optional

from ..specs.framework_spec import FrameworkSpec
from ..specs.model_spec import LlmModelSpec
from ..specs.system_spec import SystemSpec
from ..specs.partition_spec import PartitionSpec
from ..specs.tuner_spec import TuningSpec
from ..core.memory_model import compute_memory, MemoryResults
from ..core.decode_model import (
    compute_flops, FlopsResults,
    compute_traffic, TrafficResults,
    compute_comm, CommResults,
    compute_latency, LatencyResults,
)


@dataclass
class InferenceResults:
    memory: MemoryResults
    flops: FlopsResults
    traffic: TrafficResults
    comm: CommResults
    latency: LatencyResults


class InferenceCalculator:
    """High-level façade for inference performance modeling.

    Five-spec composition (`model × system × partition × tuner × framework`).
    `framework` is optional; when omitted, defaults to `FrameworkSpec.default()`
    (neutral / no-overhead — pure roofline). Pass an explicit framework to
    model production stack behavior; load via `load_framework_from_db("dynamo_trt")`
    or construct ad-hoc with `FrameworkSpec(...)`.
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

    def run(self) -> InferenceResults:
        memory = compute_memory(self.model, self.system, self.partition, self.tuner, self.framework)
        flops = compute_flops(self.model, self.partition, self.tuner, self.framework)
        traffic = compute_traffic(self.model, self.partition, self.tuner, self.framework)
        comm = compute_comm(self.model, self.system, self.partition, self.tuner, self.framework)
        latency = compute_latency(self.model, self.system, self.partition, self.tuner, self.framework, flops, traffic, comm)
        return InferenceResults(
            memory=memory,
            flops=flops,
            traffic=traffic,
            comm=comm,
            latency=latency,
        )
