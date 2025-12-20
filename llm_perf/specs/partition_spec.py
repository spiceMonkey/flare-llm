from dataclasses import dataclass

@dataclass
class PartitionSpec:
    """
    Parallel partitioning of the model across devices.
    Purely describes how we shard: PP, TP, EP, SP.
    DP is inferred from the total number of devices available.
    """

    PP: int
    TP: int
    EP: int
    SP: int