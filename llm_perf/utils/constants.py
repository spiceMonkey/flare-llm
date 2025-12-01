"""Project-wide numeric and enum constants.

This module centralizes small, reusable constants so they can be
imported instead of hardcoding magic values across the codebase.
"""

# Algorithm choices for tensor/experts parallel collectives
TP_ALGORITHMS = ("ring", "tree")
EP_ALGORITHMS = ("ring", "tree")

# Unit scaling factors
GB_TO_BYTES = 1e9          # decimal gigabytes to bytes
TB_TO_FLOPS = 1e12         # tera-FLOPs to FLOPs
US_TO_SECONDS = 1e-6       # microseconds to seconds

__all__ = [
    "TP_ALGORITHMS",
    "EP_ALGORITHMS",
    "GB_TO_BYTES",
    "TB_TO_FLOPS",
    "US_TO_SECONDS",
]
