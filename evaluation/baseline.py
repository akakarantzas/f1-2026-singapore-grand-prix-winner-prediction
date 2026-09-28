"""Freeze and verify the baseline without modifying any published forecast."""
import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).with_name("experiment.json")
BASELINE = Path(__file__).parent / "baselines" / "singapore-v1"
ARTIFACTS = ["singapore_model.pkl", "singapore_predictions.json", "singapore_metadata.json",
             "singapore_inference.json", "requirements.lock.txt", "train_singapore.py"]


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def file_hash(path):
    # Git may convert text line endings across platforms; hash semantic JSON and LF text.
    if path.suffix == ".json":
        raw = encoded(json.loads(path.read_text(encoding="utf-8"))).encode()
    elif path.suffix in {".txt", ".py"}:
        raw = path.read_text(encoding="utf-8").replace("\r\n", "\n").encode()
    else:
        raw = path.read_bytes()
    return hashlib.sha256(raw).hexdigest()


def verify(directory=BASELINE):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["artifacts"].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
    if file_hash(CONFIG) != manifest["experiment_hash"]:
        raise ValueError("Experiment rules changed; create a new experiment version")
    import joblib
    import pandas as pd
    import sklearn
    from threadpoolctl import threadpool_limits
    bundle = json.loads((directory / "singapore_inference.json").read_text())
    predictions = json.loads((directory / "singapore_predictions.json").read_text())
    metadata = json.loads((directory / "singapore_metadata.json").read_text())
    if digest([predictions, metadata]) != bundle["base_hash"]:
        raise ValueError("Frozen feature bundle baseline mismatch")
    if file_hash(directory / "singapore_model.pkl") != bundle["model_sha256"]:
        raise ValueError("Frozen model checksum mismatch")
    if sklearn.__version__ != bundle["sklearn_version"]:
        raise ValueError("Use the pinned scikit-learn version")
    rows = pd.DataFrame(bundle["rows"])
    with threadpool_limits(limits=2):
        raw = joblib.load(directory / "singapore_model.pkl").predict_proba(rows[bundle["features"]])[:, 1]
    points, teams = rows.AvgPoints5.clip(lower=0), rows.TeamAvgPoints5.clip(lower=0)
    prior = (1 / rows.GridPosition.clip(lower=1) + points / (points.max() or 1)
             + teams / (teams.max() or 1) + rows.WinRate10.clip(lower=0)
             + rows[bundle["circuit_win_rate_feature"]].clip(lower=0))
    config = metadata["prediction_postprocess"]
    probs = config["model_weight"] * raw + (1 - config["model_weight"]) * prior.to_numpy() / prior.sum() + config["floor_before_normalization"]
    probs /= probs.sum()
    reproduced = sorted([{"driver": row["driver"], "team": row["TeamName"], "probability": round(float(p), 4)}
                         for row, p in zip(bundle["rows"], probs)], key=lambda p: p["probability"], reverse=True)
    if reproduced != predictions:
        raise ValueError("Baseline probability parity failed")
    return {"status": "verified", "drivers": len(predictions), "experiment_hash": manifest["experiment_hash"]}


def freeze(revision=None):
    if (BASELINE / "manifest.json").exists():
        return verify()
    revision = revision or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Provide a full baseline Git revision")
    BASELINE.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACTS:
        if (BASELINE / name).exists() and file_hash(BASELINE / name) != file_hash(ROOT / name):
            raise ValueError("Partial baseline differs from current artifacts")
        shutil.copyfile(ROOT / name, BASELINE / name)
    import train_singapore
    manifest = {"schema_version": 1, "baseline_revision": revision,
                "experiment_hash": file_hash(CONFIG), "python_version": platform.python_version(),
                "artifacts": {name: file_hash(BASELINE / name) for name in ARTIFACTS},
                "race_universe": train_singapore.RACES_TO_LOAD,
                "forecast_kind": "published_pre_qualifying_baseline",
                "historical_2026_role": "already_observed_retrospective_evaluation"}
    (BASELINE / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--revision")
    args = parser.parse_args()
    print(json.dumps(freeze(args.revision) if args.freeze else verify(), indent=2))
