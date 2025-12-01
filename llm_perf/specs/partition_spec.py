from dataclasses import dataclass

@dataclass
class PartitionSpec:
    """
    Parallel partitioning of the model across devices.
    Purely describes how we shard: DP, PP, TP, EP, SP.
    """

    DP: int
    PP: int
    TP: int
    EP: int
    SP: int