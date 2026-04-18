
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
        "t_stage": {
            "description": "Per-PP-stage step time with overlap (pre-bubble).",
            "latex": r"t_{\text{stage}} = t_{\text{local}} + \max(0, t_{\text{comm}} - \rho\, t_{\text{local}})",
            "expr": " t_local + max(0, t_comm - rho * t_local) ",
        },
        "pp_bubble_factor": {
            "description": "Pipeline-bubble multiplier for user-observed step time.",
            "latex": r"\text{pp\_bubble\_factor} = \max\!\left(1,\; \frac{PP}{B}\right)",
            "expr": " max(1.0, PP / max(1, B)) ",
        },
        "t_step_user": {
            "description": "User-observed per-step decode time (bubble-corrected).",
            "latex": r"t_{\text{step,user}} = t_{\text{stage}} \cdot \max\!\left(1,\; \frac{PP}{B}\right)",
            "expr": " t_stage * max(1.0, PP / max(1, B)) ",
        },
        "TPOT": {
            "description": "Time per output token (user-observed): equals step time, not t_stage/B.",
            "latex": r"\text{TPOT}(B) = t_{\text{step,user}}(B) = t_{\text{stage}}(B) \cdot \max\!\left(1,\; \frac{PP}{B}\right)",
            "expr": " t_step_user ",
        },
        "TPS_single": {
            "description": "Per-DP-replica decode throughput.",
            "latex": r"TPS_{\text{single}} = \frac{B}{t_{\text{step,user}}}",
            "expr": " B / t_step_user ",
        },
        "TTPS": {
            "description": "Global decode throughput across DP replicas.",
            "latex": r"TTPS = \frac{DP \cdot B}{t_{\text{step,user}}}",
            "expr": " DP * B / t_step_user ",
        },
        "msg_PP": {
            "description": "Per-device PP hop payload per decode step (B activations).",
            "latex": r"m_{PP} = B \cdot (H/TP) \cdot b",
            "expr": " B * (H/TP) * b ",
        },
        "msg_TP": {
            "description": "Per-device TP all-reduce payload per decode step.",
            "latex": r"m_{TP} = B \cdot H \cdot b",
            "expr": " B * H * b ",
        },
        "msg_EP": {
            "description": "Per-device EP all-to-all payload per decode step (MoE).",
            "latex": r"m_{EP} = B \cdot k \cdot H \cdot b",
            "expr": " B * k * H * b ",
        },
        "msg_SP": {
            "description": "Per-device SP ring all-gather payload per decode step.",
            "latex": r"m_{SP} = B \cdot (S/SP) \cdot (2 H_{kv}/TP) \cdot b",
            "expr": " B * (S/SP) * (2*H_kv/TP) * b ",
        },
        "B_star": {
            "description": "Compute-bound crossover batch size.",
            "latex": r"B^* = \frac{T_\theta \cdot R_{\text{GPU}}}{F_{\text{token}} \cdot B_{\text{eff,mem}} - T_{\text{KV}} \cdot R_{\text{GPU}}}",
            "expr": " T_theta * R_gpu / (F_token * B_eff_mem - T_kv * R_gpu) ",
        },
        "F_prefill_device": {
            "description": "Per-device prefill FLOPs, split by dense vs MoE layers. MoE adds an unsharded router term.",
            "latex": r"F_{\text{prefill,device}} = \frac{L_{\text{dense}}}{PP}\left[\frac{(4H^2 + 4HH_{kv} + 6HI_{\text{dense}})S}{TP} + \frac{4S^2 H}{TP \cdot SP}\right] + \frac{L_{\text{moe}}}{PP}\left[\frac{(4H^2 + 4HH_{kv})S}{TP} + \frac{6HkI_{\text{moe}}S}{TP \cdot EP} + \frac{4S^2 H}{TP \cdot SP} + 2HN_{\text{exp}}S\right]",
            "expr": " (L_dense/PP) * ((4*H**2+4*H*H_kv+6*H*I_dense)*S/TP + 4*S**2*H/(TP*SP)) + (L_moe/PP) * ((4*H**2+4*H*H_kv)*S/TP + 6*H*k*I_moe*S/(TP*EP) + 4*S**2*H/(TP*SP) + 2*H*N_exp*S) ",
        },
        "F_router_prefill": {
            "description": "Per-MoE-layer prefill router FLOPs (unsharded across TP; zero for dense layers).",
            "latex": r"F_{\text{router,prefill}} = 2 H N_{\text{exp}} S_{\text{input}}",
            "expr": " 2 * H * N_exp * S_input ",
        },
        "t_prefill": {
            "description": "Hardware prefill latency (single-request, co-located).",
            "latex": r"t_{\text{prefill}} = t_{\text{prefill,local}} + \max(0, t_{\text{prefill,comm}} - \rho\, t_{\text{prefill,local}}) + t_{\text{pipeline,warmup}}",
            "expr": " t_prefill_local + max(0, t_prefill_comm - rho * t_prefill_local) + t_pipeline_warmup ",
        },
        "TTFT_single": {
            "description": "Time to first token (co-located single-request).",
            "latex": r"TTFT = t_{\text{sched}} + t_{\text{prefill}} + t_{\text{step,user}}",
            "expr": " t_sched + t_prefill + t_step_user ",
        },
        "TTFT_disagg": {
            "description": "TTFT for disaggregated prefill architecture.",
            "latex": r"TTFT_{\text{disagg}} = t_{\text{sched}} + t_{\text{prefill}} + t_{\text{KV,transfer}} + t_{\text{step,user}}",
            "expr": " t_sched + t_prefill + t_KV_transfer + t_step_user ",
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
