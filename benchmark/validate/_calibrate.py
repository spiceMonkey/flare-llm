#!/usr/bin/env python3
"""One-shot helper to find best-fit (bw_eta, c_serving) for each driver.

Imports each driver's MODEL/SYSTEM/PRECISION/ISL/OSL constants (and the
load_measured filters) and runs a small grid sweep. Used to populate the
DEFAULT_BW_ETA / DEFAULT_C_SERVING_US constants at the top of each driver.

Run periodically (e.g. when refresh fetch.py pulls new InferenceX rows) to
re-tune. Not part of the regular validator workflow.
"""
import dataclasses
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    InferenceCalculator, PartitionSpec, TuningSpec,
    load_measured, system_with_eta,
    load_model_from_db, load_system_from_db,
)
from llm_perf.specs import FrameworkSpec  # noqa: E402


def calibrate(
    *,
    model_id: str,
    system_id: str,
    bytes_per_param: float | None,
    PP: int, TP: int, EP: int, SP: int,
    attention_mode: str, layout: str,
    num_devices: int,
    measured_loader,  # callable returning list of MeasuredPoint
    S_decode_fn,      # callable (m: MeasuredPoint) -> S_decode for that row
):
    """Sweep (bw_eta, c_serving) and return the best-fit tuple."""
    measured = measured_loader()
    if not measured:
        return None
    m = load_model_from_db(model_id)
    if bytes_per_param is not None:
        m = dataclasses.replace(m, bytes_per_param=bytes_per_param)

    best = (1e9, None)
    for bw in (1.0, 0.85, 0.7, 0.55, 0.4):
        for cs in (0, 5, 10, 22, 50, 75, 100):
            errs = []
            for row in measured:
                s = system_with_eta(load_system_from_db(system_id),
                                    num_devices=num_devices, bw_eta=bw)
                p = PartitionSpec(PP=PP, TP=TP, EP=EP, SP=SP)
                t = TuningSpec(S_decode=S_decode_fn(row), B_decode=row.B)
                fw = FrameworkSpec(name="calibrate", c_serving_per_seq_us=cs,
                                   attention_mode=attention_mode, layout=layout)
                r = InferenceCalculator(m, s, p, t, fw).run()
                errs.append((r.latency.TPOT * 1000 - row.tpot_ms) / row.tpot_ms * 100)
            mae = float(np.mean(np.abs(errs)))
            if mae < best[0]:
                best = (mae, (bw, cs))
    return best


def main():
    s_decode = lambda m: m.isl + m.osl // 2

    # Each entry: (label, model_id, system_id, bytes/param,
    #              PP/TP/EP/SP, attn_mode, layout, num_devices, measured filter)
    cuts = [
        ("dsr1_gb200_dynamo_trt — colocated TP=EP=8",
         "deepseek_r1_0528", "gb200.72gpu", 0.5, (1, 8, 8, 1), "dp", "co_located", 8,
         lambda: load_measured("DeepSeek-R1-0528", isl=1024, osl=1024, precision="fp4",
                               decode_tp=8, decode_ep=8, num_decode_gpu=8,
                               dp_attention=True, framework={"dynamo-trt", "trt", "trt-llm", "trtllm", "dynamo-trt-llm"},
                               hardware="gb200")),
        ("dsr1_b200_trt — TP=8 EP=1 dec=8",
         "deepseek_r1_0528", "b200.8gpu", 0.5, (1, 8, 1, 1), "tp", "orthogonal", 8,
         lambda: load_measured("DeepSeek-R1-0528", isl=1024, osl=1024, precision="fp4",
                               decode_tp=8, decode_ep=1, num_decode_gpu=8,
                               framework={"trt", "trt-llm", "trtllm"},
                               hardware="b200")),
        ("dsr1_gb300_dynamo_sglang — colocated TP=EP=32",
         "deepseek_r1_0528", "gb300.72gpu", 0.5, (1, 32, 32, 1), "dp", "co_located", 32,
         lambda: load_measured("DeepSeek-R1-0528", isl=1024, osl=1024, precision="fp4",
                               decode_tp=32, decode_ep=32, num_decode_gpu=32,
                               dp_attention=True, framework={"dynamo-sglang", "sglang"},
                               hardware="gb300")),
        ("gpt_oss_120b_gb200_dynamo_trt — TP=4 EP=1 dec=4",
         "gpt_oss_120b", "gb200.72gpu", None, (1, 4, 1, 1), "tp", "orthogonal", 4,
         lambda: load_measured("gpt-oss-120b", isl=1024, osl=1024, precision="fp4",
                               decode_tp=4, decode_ep=1, num_decode_gpu=4,
                               framework={"dynamo-trt", "trt", "trt-llm", "trtllm", "dynamo-trt-llm"},
                               hardware="gb200")),
        ("gpt_oss_120b_h200_trt — TP=4 EP=1 dec=4",
         "gpt_oss_120b", "h200.8gpu", None, (1, 4, 1, 1), "tp", "orthogonal", 4,
         lambda: load_measured("gpt-oss-120b", isl=1024, osl=1024, precision="fp4",
                               decode_tp=4, decode_ep=1, num_decode_gpu=4,
                               framework={"trt", "trt-llm", "trtllm"},
                               hardware="h200")),
        ("llama3_70b_b200_trt — TP=4",
         "llama3.1_70b", "b200.8gpu", 1, (1, 4, 1, 1), "tp", "orthogonal", 4,
         lambda: load_measured("Llama-3.3-70B-Instruct-FP8", isl=1024, osl=1024, precision="fp8",
                               decode_tp=4, decode_ep=1, num_decode_gpu=4,
                               framework={"trt", "trt-llm", "trtllm"},
                               hardware="b200")),
        ("llama3_70b_h200_trt — TP=4",
         "llama3.1_70b", "h200.8gpu", 1, (1, 4, 1, 1), "tp", "orthogonal", 4,
         lambda: load_measured("Llama-3.3-70B-Instruct-FP8", isl=1024, osl=1024, precision="fp8",
                               decode_tp=4, decode_ep=1, num_decode_gpu=4,
                               framework={"trt", "trt-llm", "trtllm"},
                               hardware="h200")),
    ]

    print(f"{'driver / cut':<60} {'best (bw, c_serv)':>18} {'MAE':>7}")
    print("-" * 90)
    for label, model_id, sys_id, bpp, (PP, TP, EP, SP), attn, layout, nd, ld in cuts:
        result = calibrate(
            model_id=model_id, system_id=sys_id, bytes_per_param=bpp,
            PP=PP, TP=TP, EP=EP, SP=SP,
            attention_mode=attn, layout=layout, num_devices=nd,
            measured_loader=ld, S_decode_fn=s_decode,
        )
        if result is None:
            print(f"{label:<60} {'(no measured)':>18} {'-':>7}")
            continue
        mae, (bw, cs) = result
        print(f"{label:<60} {f'({bw}, {cs})':>18} {mae:>6.1f}%")


if __name__ == "__main__":
    main()
