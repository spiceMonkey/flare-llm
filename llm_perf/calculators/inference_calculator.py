
from dataclasses import dataclass
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
    """High-level façade for inference performance modeling."""

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

    def run(self) -> InferenceResults:
        memory = compute_memory(self.model, self.system, self.partition, self.tuner)
        flops = compute_flops(self.model, self.partition, self.tuner)
        traffic = compute_traffic(self.model, self.partition, self.tuner)
        comm = compute_comm(self.model, self.system, self.partition, self.tuner)
        latency = compute_latency(self.model, self.system, self.partition, self.tuner, flops, traffic, comm)
        return InferenceResults(
            memory=memory,
            flops=flops,
            traffic=traffic,
            comm=comm,
            latency=latency,
        )
