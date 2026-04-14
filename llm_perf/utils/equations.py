
"""Static registry of canonical equations (LaTeX + Python literal strings)."""


class LlmPerfEquations:
    EQUATIONS = {
        "M_theta_device": {
            "description": "Per-device parameter memory.",
            "latex": r"M_{\theta,\text{device}} = \frac{L}{PP}\left(\frac{2H^2 + 2 H H_{kv}}{TP} + \frac{3 H I N_{\text{exp}}}{TP \cdot EP}\right) b + \frac{V H}{TP} b",
            "expr": " (L/PP) * ((2*H**2 + 2*H*H_kv)/TP + (3*H*I*N_exp)/(TP*EP)) * b + (V*H/TP)*b ",
        },
        "M_act_device": {
            "description": "Per-device activation memory.",
            "latex": r"M_{\text{act,device}} = (4H + 2H_{kv}) b",
            "expr": " (4*H + 2*H_kv) * b ",
        },
        "M_kv_device": {
            "description": "Per-device KV cache memory.",
            "latex": r"M_{\text{KV,device}} = \frac{L}{PP} \frac{2 S H_{kv} b}{TP \cdot SP}",
            "expr": " (L/PP) * (2*S*H_kv*b) / (TP*SP) ",
        },
        "F_token_device": {
            "description": "Per-device FLOPs per decoded token on a PP stage.",
            "latex": r"F_{\text{token,device}} \approx \frac{L}{PP}\left(\frac{4H^2 + 4 H H_{kv}}{TP} + \frac{6H I_{\text{eff}}}{TP \cdot EP} + \frac{4 S H}{TP \cdot SP} + 2H N_{\text{eff}}\right)",
            "expr": (
                " (L/PP) * ("
                " (4*H**2 + 4*H*H_kv)/TP"
                " + (6*H*I_eff)/(TP*EP)"
                " + (4*S*H)/(TP*SP)"
                " + 2*H*N_eff"
                ") "
            ),
        },
        "t_token": {
            "description": "Per-token latency with overlap.",
            "latex": r"t_{\text{token}} = t_{\text{local}} + \max(0, t_{\text{comm}} - \rho t_{\text{local}})",
            "expr": " t_local + max(0, t_comm - rho * t_local) ",
        },
        "TPOT": {
            "description": "Time per output token (batched decode).",
            "latex": r"\text{TPOT} = \frac{t_{\text{token}}(B)}{B}",
            "expr": " t_token / B ",
        },
        "B_star": {
            "description": "Compute-bound crossover batch size.",
            "latex": r"B^* = \frac{T_\theta \cdot R_{\text{GPU}}}{F_{\text{token}} \cdot B_{\text{eff,mem}} - T_{\text{KV}} \cdot R_{\text{GPU}}}",
            "expr": " T_theta * R_gpu / (F_token * B_eff_mem - T_kv * R_gpu) ",
        },
        "F_prefill_device": {
            "description": "Per-device prefill FLOPs.",
            "latex": r"F_{\text{prefill,device}} = \frac{L}{PP}\left[\frac{(4H^2 + 4HH_{kv})S}{TP} + \frac{6HI_{\text{eff}}S}{TP \cdot EP} + \frac{4S^2 H}{TP \cdot SP}\right]",
            "expr": " (L/PP) * ((4*H**2+4*H*H_kv)*S/TP + 6*H*I_eff*S/(TP*EP) + 4*S**2*H/(TP*SP)) ",
        },
        "t_prefill": {
            "description": "Hardware prefill latency (single-request, co-located).",
            "latex": r"t_{\text{prefill}} = t_{\text{prefill,local}} + \max(0, t_{\text{prefill,comm}} - \rho\, t_{\text{prefill,local}}) + t_{\text{pipeline,warmup}}",
            "expr": " t_prefill_local + max(0, t_prefill_comm - rho * t_prefill_local) + t_pipeline_warmup ",
        },
        "TTFT_single": {
            "description": "Time to first token (co-located single-request).",
            "latex": r"TTFT = t_{\text{sched}} + t_{\text{prefill}} + t_{\text{token}}",
            "expr": " t_sched + t_prefill + t_token ",
        },
        "TTFT_disagg": {
            "description": "TTFT for disaggregated prefill architecture.",
            "latex": r"TTFT_{\text{disagg}} = t_{\text{sched}} + t_{\text{prefill}} + t_{\text{KV,transfer}} + t_{\text{token}}",
            "expr": " t_sched + t_prefill + t_KV_transfer + t_token ",
        },
        "E2E_latency": {
            "description": "End-to-end request latency.",
            "latex": r"E2E(N_{\text{out}}) = TTFT + (N_{\text{out}} - 1) \times \text{TPOT}",
            "expr": " TTFT + (N_out - 1) * TPOT ",
        },
        "S_max_paged": {
            "description": "Max context length with paged KV cache.",
            "latex": r"S_{\max} = \frac{M_{\text{HBM}} - M_\theta - M_{\text{act}} - M_{\text{sys}}}{2 H_{kv} b (L/PP) / (TP \cdot SP)}",
            "expr": " M_avail / (2 * H_kv * b * (L/PP) / (TP * SP)) ",
        },
        "BW_dram3d": {
            "description": "DRAM3D bandwidth from physical parameters.",
            "latex": r"BW = n_{\text{dies}} \times \lfloor A_{\text{die}} / p^2 \rfloor \times \eta_{\text{data}} \times f_{\text{data}} / 8",
            "expr": " n_dies * int(die_area_um2 / pitch_um**2) * data_pin_fraction * data_rate_gbps / 8 ",
        },
    }

    @classmethod
    def list_ids(cls):
        return sorted(cls.EQUATIONS.keys())

    @classmethod
    def get(cls, eq_id: str):
        return cls.EQUATIONS.get(eq_id)

    @classmethod
    def latex(cls, eq_id: str):
        meta = cls.get(eq_id)
        return meta["latex"] if meta else None

    @classmethod
    def expr(cls, eq_id: str):
        meta = cls.get(eq_id)
        return meta["expr"] if meta else None
