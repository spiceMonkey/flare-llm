"""Per-pipeline-stage communication aggregator.

Local helper extracted from the upstream-synced ``collective_cost.py``;
the upstream module no longer carries llm_perf-specific glue (every
function there is a pure α-β collective primitive). This file owns the
TP/SP/EP/PP combination logic — see ``documentation/modeling/decode.md``
§5.5 for the derivation and ``documentation/modeling/prefill.md`` §3.2
for the prefill cross-reference.
"""


def aggregate_per_stage(
    L: int,
    L_moe: int,
    PP: int,
    n_TP: int,
    t_TP: float,
    n_SP: int,
    t_SP: float,
    n_EP: int,
    t_EP: float,
    t_PP: float,
) -> float:
    """Per-pipeline-stage communication time — decode.md §5.5.

        t = (L/PP)·(n_TP·t_TP + n_SP·t_SP)
          + (L_moe/PP)·(n_EP·t_EP)
          + t_PP

    TP and SP collectives apply to every layer on the stage.
    EP collectives apply only to MoE layers (L_moe/PP per stage).
    The PP hop is a single inter-stage forward per step.

    Prefill cross-references this aggregation — see
    ``documentation/modeling/prefill.md`` §3.2 ("Following the same
    structure as decode.md §5.5").
    """
    return (L / PP) * (n_TP * t_TP + n_SP * t_SP) + (L_moe / PP) * (n_EP * t_EP) + t_PP
