"""Side-by-side E2E demo: co-located vs. disaggregated prefill/decode.

Loads a model/system/partition/tuner, runs Inference + Prefill + E2E for a
co-located deployment (no KV handoff) and for a disaggregated deployment
(calibrated MoonCake-class parameters), and prints a TTFT / TPOT / throughput
card for both.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_perf import (
    DisaggSpec,
    InferenceCalculator,
    OverheadSpec,
    PartitionSpec,
)
from llm_perf.calculators.e2e_calculator import E2ECalculator
from llm_perf.calculators.prefill_calculator import PrefillCalculator
from llm_perf.io import load_model_spec, load_system_spec, load_tuning_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="llm_perf/database/model/qwen3_vl_235b_fp8.json",
        type=Path,
    )
    parser.add_argument(
        "--system",
        default="llm_perf/database/system/h100.8gpu.json",
        type=Path,
    )
    parser.add_argument(
        "--tuner",
        default="llm_perf/database/tuner/qwen3.tuner.json",
        type=Path,
    )
    parser.add_argument("--PP", type=int, default=1)
    parser.add_argument("--TP", type=int, default=8)
    parser.add_argument("--EP", type=int, default=4)
    parser.add_argument("--SP", type=int, default=1)
    parser.add_argument("--S-input", type=int, default=4096, help="Prefill sequence length")
    parser.add_argument("--S-decode", type=int, default=4096, help="Decode effective context")
    parser.add_argument("--B-decode", type=int, default=1)
    parser.add_argument(
        "--rho-KV",
        type=float,
        default=0.8,
        help="Layer-wise streaming overlap factor in [0,1] for the disagg case",
    )
    parser.add_argument(
        "--BW-inter",
        type=float,
        default=50.0,
        help="Effective delivered inter-cluster bandwidth GB/s (ConnectX-7 ~50)",
    )
    parser.add_argument(
        "--BW-intra",
        type=float,
        default=900.0,
        help="Scale-up fabric bandwidth GB/s (H100 NVLink ~900)",
    )
    return parser.parse_args()


def fmt_ms(s: float) -> str:
    return f"{s * 1e3:8.3f} ms"


def fmt_us(s: float) -> str:
    return f"{s * 1e6:8.2f} µs"


def print_card(label: str, r, t_prefill: float) -> None:
    print(f"\n── {label} " + "─" * (60 - len(label)))
    print(f"  t_sched            {fmt_us(r.t_sched)}")
    print(f"  t_prefill          {fmt_ms(t_prefill)}")
    print(f"  t_handoff          {fmt_ms(r.t_handoff)}")
    print(f"    └─ t_repack      {fmt_us(r.t_repack)}")
    print(f"  t_first_token      {fmt_us(r.TTFT - r.t_sched - t_prefill - r.t_handoff)}")
    print(f"  ─────────────────────────────")
    print(f"  TTFT               {fmt_ms(r.TTFT)}")
    print(f"  TPOT               {fmt_us(r.TPOT)}")
    print(f"  TTPS               {r.TTPS:8.2f} tok/s")
    print(f"  throughput/GPU     {r.throughput_per_gpu:8.2f} tok/s")
    print(f"  interactivity      {r.interactivity:8.2f} tok/s (1/TPOT)")
    print(f"  M_KV_total         {r.M_KV_total / 1e9:8.3f} GB")


def main() -> None:
    args = parse_args()

    model = load_model_spec(args.model)
    system = load_system_spec(args.system)
    tuner = load_tuning_spec(args.tuner)
    tuner.S_decode = args.S_decode
    tuner.S_input = args.S_input
    tuner.B_decode = args.B_decode

    partition = PartitionSpec(PP=args.PP, TP=args.TP, EP=args.EP, SP=args.SP)

    dec = InferenceCalculator(model, system, partition, tuner).run()
    pre = PrefillCalculator(model, system, partition, tuner).run()
    t_prefill = pre.latency.t_prefill

    print("=" * 64)
    print(f" Model:      {model.name}")
    print(f" System:     {system.name}  ({system.num_devices} devices)")
    print(f" Partition:  PP={args.PP} TP={args.TP} EP={args.EP} SP={args.SP}")
    print(f" S_input:    {args.S_input:,}  S_decode: {args.S_decode:,}  B: {args.B_decode}")
    print("=" * 64)

    oh = OverheadSpec()

    # Co-located deployment: prefill and decode share the cluster and the
    # partition is identical (no repack, no handoff)
    disagg_colo = DisaggSpec(disaggregated=False)
    r_colo = E2ECalculator(dec, pre, oh, disagg_colo, model, system, partition, tuner).run()
    print_card("Co-located (matched partition, zero handoff)", r_colo, t_prefill)

    # Disaggregated deployment: MoonCake-class parameters
    disagg_d = DisaggSpec(
        disaggregated=True,
        inter_alpha_us=5.0,
        inter_bandwidth_GBps=args.BW_inter,
        N_WR=model.L * args.TP * args.SP,
        tau_WR_us=1.0,
        overlap_rho_KV=args.rho_KV,
        repack_GBps=args.BW_intra,
        repack_eta=1.2,
    )
    r_d = E2ECalculator(dec, pre, oh, disagg_d, model, system, partition, tuner).run()
    print_card(
        f"Disaggregated (BW_inter={args.BW_inter:.0f} GB/s, ρ_KV={args.rho_KV})",
        r_d, t_prefill,
    )

    # Delta
    ttft_delta = r_d.TTFT - r_colo.TTFT
    print("\n── Summary " + "─" * 54)
    print(f"  Δ TTFT (disagg − colo): {fmt_ms(ttft_delta)}")
    print(f"  Disagg handoff budget   {fmt_ms(r_d.t_handoff)}")
    print(f"    (raw transfer {fmt_ms(r_d.M_KV_total / (args.BW_inter * 1e9))}, ρ_KV absorbs {args.rho_KV:.0%} of t_prefill = {fmt_ms(args.rho_KV * t_prefill)})")


if __name__ == "__main__":
    main()
