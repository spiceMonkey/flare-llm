
"""Static registry of canonical equations (LaTeX + Python literal strings)."""


class LlmPerfEquations:
    EQUATIONS = {
        "M_theta_device": {
            "description": "Per-device parameter memory.",
            "latex": r"M_{\theta,\text{device}} = \frac{L}{PP}\left(\frac{H^2 + 3 H H_{kv}}{TP} + \frac{2 H I N_{\text{exp}}}{TP \cdot EP}\right) b + \frac{V H}{TP} b",
            "expr": " (L/PP) * ((H**2 + 3*H*H_kv)/TP + (2*H*I*N_exp)/(TP*EP)) * b + (V*H/TP)*b ",
        },
        "M_act_device": {
            "description": "Per-device activation memory.",
            "latex": r"M_{\text{act,device}} = \frac{L}{PP}(4H + 2H_{kv}) b",
            "expr": " (L/PP) * (4*H + 2*H_kv) * b ",
        },
        "M_kv_device": {
            "description": "Per-device KV cache memory.",
            "latex": r"M_{\text{KV,device}} = \frac{L}{PP} \frac{2 S H_{kv} b}{TP \cdot SP}",
            "expr": " (L/PP) * (2*S*H_kv*b) / (TP*SP) ",
        },
        "F_token_device": {
            "description": "Per-device FLOPs per decoded token on a PP stage.",
            "latex": r"F_{\text{token,device}} \approx \frac{L}{PP}\left(\frac{2H^2 + 6 H H_{kv}}{TP} + \frac{4H I_{\text{eff}}}{TP \cdot EP} + \frac{4 S H_{kv}}{TP \cdot SP} + 2H N_{\text{eff}}\right)",
            "expr": (
                " (L/PP) * ("
                " (2*H**2 + 6*H*H_kv)/TP"
                " + (4*H*I_eff)/(TP*EP)"
                " + (4*S*H_kv)/(TP*SP)"
                " + 2*H*N_eff"
                ") "
            ),
        },
        "t_token": {
            "description": "Per-token latency with overlap.",
            "latex": r"t_{\text{token}} \approx \max(t_{\text{local}}, \rho t_{\text{comm}})",
            "expr": " max(t_local, rho * t_comm) ",
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
