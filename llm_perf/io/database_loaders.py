# llm_perf/io/database_loaders.py

from pathlib import Path
from typing import List

from .model_loaders import load_model_spec
from .system_loaders import load_system_spec
from .partition_loaders import load_partition_spec
from .tuner_loaders import load_tuning_spec

# Base database dir: llm_perf/database
_DB_ROOT = Path(__file__).resolve().parent.parent / "database"

_HW_DIR = _DB_ROOT / "system"
_MODEL_DIR = _DB_ROOT / "model"
_PARTITION_DIR = _DB_ROOT / "partition"
_TUNER_DIR = _DB_ROOT / "tuner"

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

def list_partition_ids() -> List[str]:
    """
    List available partition IDs from llm_perf/database/partition.
    """
    if not _PARTITION_DIR.is_dir():
        return []
    return sorted(p.stem for p in _PARTITION_DIR.glob("*.json"))


def load_partition_from_db(partition_id: str):
    """
    Load a PartitionSpec from llm_perf/database/partition/{partition_id}.json.
    """
    path = _PARTITION_DIR / f"{partition_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"No partition config found for id={partition_id!r} at {path}")
    return load_partition_spec(path)


# ─────────────────────────────────────────────
# Tuners (S_decode, flash_attn_gain, overlap, algorithms, etc.)
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
