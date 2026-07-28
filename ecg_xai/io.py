import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple


def save_feature_set(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    names: list[str],
    family: Dict[str, str],
    band_map: Dict[str, str],
    filepath: str | Path,
    compress: bool = True
) -> None:
    """
    Save the complete engineered feature set with metadata.
    
    Recommended for reproducibility and faster experimentation.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "X": X.astype(np.float32),
        "y": y,
        "groups": groups,
        "names": np.array(names, dtype=object),
        "family": json.dumps(family),
        "band_map": json.dumps(band_map),
    }

    if compress:
        np.savez_compressed(filepath, **save_dict)
    else:
        np.savez(filepath, **save_dict)

    print(f"Saved engineered features to: {filepath} "
          f"(shape={X.shape}, size={filepath.stat().st_size / 1024:.1f} KB)")


def load_feature_set(filepath: str | Path) -> Tuple:
    """
    Load saved engineered feature set.
    Returns: X, y, groups, names, family, band_map
    """
    filepath = Path(filepath)
    data = np.load(filepath, allow_pickle=True)

    X = data["X"]
    y = data["y"]
    groups = data["groups"]
    names = data["names"].tolist()
    family = json.loads(str(data["family"]))
    band_map = json.loads(str(data["band_map"]))

    return X, y, groups, names, family, band_map