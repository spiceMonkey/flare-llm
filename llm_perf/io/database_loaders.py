# llm_perf/io/database_loaders.py

from pathlib import Path
from typing import List

from .framework_loaders import load_framework_spec
from .model_loaders import load_model_spec
from .system_loaders import load_system_spec
from .partition_loaders import load_partition_spec
from .tuner_loaders import load_tuning_spec

# Base database dir: llm_perf/database
_DB_ROOT = Path(__file__).resolve().parent.parent / "database"

_HW_DIR = _DB_ROOT / "system"
_MODEL_DIR = _DB_ROOT / "model"
_TUNER_DIR = _DB_ROOT / "tuner"
_FRAMEWORK_DIR = _DB_ROOT / "framework"

# ─────────────────────────────────────────────
# HW systems
# ─────────────────────────────────────────────

def list_hw_system_ids() -> List[str]:
    """
    List available hardware system IDs from llm_perf/database/system.

    Returns filename stems, e.g. ["h100_node", "h100_cluster_64"].
    """
    if not _HW_DIR.is_dir():
        return []
    return sorted(p.stem for p in _HW_DIR.glob("*.json"))


def load_system_from_db(system_id: str):
    """
    Load a SystemSpec from llm_perf/database/system/{system_id}.json
    using the standard system loader.
    """
    path = _HW_DIR / f"{system_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"No system config found for id={system_id!r} at {path}")
    return load_system_spec(path)


# ─────────────────────────────────────────────
# LLM models
# ─────────────────────────────────────────────

def list_model_ids() -> List[str]:
    """
    List available LLM model IDs from llm_perf/database/model.
    """
    if not _MODEL_DIR.is_dir():
        return []
    return sorted(p.stem for p in _MODEL_DIR.glob("*.json"))


def load_model_from_db(model_id: str):
    """
    Load a LlmModelSpec from llm_perf/database/model/{model_id}.json
    using the standard model loader.
    """
    path = _MODEL_DIR / f"{model_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"No model config found for id={model_id!r} at {path}")
    return load_model_spec(path)


# ─────────────────────────────────────────────
# Partitions (DP/PP/TP/EP/SP)
# ─────────────────────────────────────────────
# Phase G: partition.json database removed. Partitions are inline-
# constructed at the call site (drivers, sweeps, notebooks) — they're
# runtime sweep parameters, not curated catalog entries. The
# `load_partition_spec` file-path loader stays in io.partition_loaders
# for ad-hoc one-off JSONs. Recommended attention_mode + tp_ep_layout per
# stack now live on FrameworkSpec (see database/framework/).
# Tuners (S_decode, B_decode, S_input, B_prefill, chunk_size, placement, n_tok_draft, p_accept)
# ─────────────────────────────────────────────

def list_tuner_ids() -> List[str]:
    """
    List available tuner IDs from llm_perf/database/tuner.
    """
    if not _TUNER_DIR.is_dir():
        return []
    return sorted(p.stem for p in _TUNER_DIR.glob("*.json"))


def load_tuner_from_db(tuner_id: str):
    """
    Load a TuningSpec from llm_perf/database/tuner/{tuner_id}.json.
    """
    path = _TUNER_DIR / f"{tuner_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"No tuner config found for id={tuner_id!r} at {path}")
    return load_tuning_spec(path)


# ─────────────────────────────────────────────
# Frameworks (SW-stack-specific runtime knobs)
# ─────────────────────────────────────────────

def list_framework_ids() -> List[str]:
    """List available framework IDs from llm_perf/database/framework."""
    if not _FRAMEWORK_DIR.is_dir():
        return []
    return sorted(p.stem for p in _FRAMEWORK_DIR.glob("*.json"))


def load_framework_from_db(framework_id: str):
    """Load a FrameworkSpec from llm_perf/database/framework/{framework_id}.json."""
    path = _FRAMEWORK_DIR / f"{framework_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"No framework config found for id={framework_id!r} at {path}")
    return load_framework_spec(path)
