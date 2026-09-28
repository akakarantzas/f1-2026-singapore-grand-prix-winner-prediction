from __future__ import annotations

import json
import hashlib
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import fastf1
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn import set_config
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")
set_config(enable_metadata_routing=True)

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", ROOT))
MODEL_PATH = OUTPUT_DIR / "singapore_model.pkl"
PREDICTIONS_PATH = OUTPUT_DIR / "singapore_predictions.json"
METADATA_PATH = OUTPUT_DIR / "singapore_metadata.json"
TARGET_START = pd.Timestamp("2026-10-11T12:00:00Z")
GRID_OVERRIDE_PATH = ROOT / "qualifying_grid.json"
POSTPROCESS_CANDIDATES = [
    {"model_weight": model_weight, "floor": floor}
    for model_weight in (0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85)
    for floor in (0.0005, 0.001, 0.002)
]

RACES_TO_LOAD = [
    (2022, "Singapore"),
    (2023, "Bahrain"),
    (2023, "Saudi Arabia"),
    (2023, "Australia"),
    (2023, "Azerbaijan"),
    (2023, "Miami"),
    (2023, "Monaco"),
    (2023, "Spain"),
    (2023, "Canada"),
    (2023, "Austria"),
    (2023, "British"),
    (2023, "Hungary"),
    (2023, "Belgium"),
    (2023, "Dutch"),
    (2023, "Italian"),
    (2023, "Singapore"),
    (2023, "Japan"),
    (2023, "Qatar"),
    (2023, "United States"),
    (2023, "Mexico"),
    (2023, "Sao Paulo"),
    (2023, "Las Vegas"),
    (2023, "Abu Dhabi"),
    (2024, "Bahrain"),
    (2024, "Saudi Arabia"),
    (2024, "Australia"),
    (2024, "Japan"),
    (2024, "China"),
    (2024, "Miami"),
    (2024, "Emilia Romagna"),
    (2024, "Monaco"),
    (2024, "Canada"),
    (2024, "Spain"),
    (2024, "Austria"),
    (2024, "British"),
    (2024, "Hungary"),
    (2024, "Belgium"),
    (2024, "Dutch"),
    (2024, "Italian"),
    (2024, "Azerbaijan"),
    (2024, "Singapore"),
    (2024, "United States"),
    (2024, "Mexico"),
    (2024, "Sao Paulo"),
    (2024, "Las Vegas"),
    (2024, "Qatar"),
    (2024, "Abu Dhabi"),
    (2025, "Australia"),
    (2025, "China"),
    (2025, "Japan"),
    (2025, "Bahrain"),
    (2025, "Saudi Arabia"),
    (2025, "Miami"),
    (2025, "Emilia Romagna"),
    (2025, "Monaco"),
    (2025, "Spain"),
    (2025, "Canada"),
    (2025, "Austria"),
    (2025, "British"),
    (2025, "Belgium"),
    (2025, "Hungary"),
    (2025, "Dutch"),
    (2025, "Italian"),
    (2025, "Azerbaijan"),
    (2025, "Singapore"),
    (2025, "United States"),
    (2025, "Mexico"),
    (2025, "Sao Paulo"),
    (2025, "Las Vegas"),
    (2025, "Qatar"),
    (2025, "Abu Dhabi"),
    (2026, "Australia"),
    (2026, "China"),
    (2026, "Japan"),
    (2026, "Miami"),
    (2026, "Canada"),
    (2026, "Monaco"),
    (2026, "Barcelona-Catalunya"),
    (2026, "Austria"),
    (2026, "British"),
    (2026, "Belgium"),
    (2026, "Hungary"),
    (2026, "Dutch"),
    (2026, "Italian"),
    (2026, "Spain"),
    (2026, "Azerbaijan"),
]

STREET_CIRCUITS = {"Miami", "Monaco", "Singapore", "Azerbaijan", "Las Vegas", "Saudi Arabia"}
FEATURES = [
    "GridPosition",
    "DriverCode",
    "TeamName",
    "AvgPoints5",
    "AvgGrid3",
    "AvgGrid5",
    "AvgFinish5",
    "WinRate10",
    "TeamAvgPoints5",
    "SingaporeExperience",
    "SingaporeWinRate",
    "IsStreetCircuit",
]
CATEGORICAL_FEATURES = ["DriverCode", "TeamName"]
NUMERIC_FEATURES = [feature for feature in FEATURES if feature not in CATEGORICAL_FEATURES]

DRIVER_ROSTER_2026 = [
    ("NOR", "Norris", "McLaren"),
    ("PIA", "Piastri", "McLaren"),
    ("RUS", "Russell", "Mercedes"),
    ("ANT", "Antonelli", "Mercedes"),
    ("VER", "Verstappen", "Red Bull Racing"),
    ("HAD", "Hadjar", "Red Bull Racing"),
    ("LEC", "Leclerc", "Ferrari"),
    ("HAM", "Hamilton", "Ferrari"),
    ("ALB", "Albon", "Williams"),
    ("SAI", "Sainz", "Williams"),
    ("LIN", "Lindblad", "Racing Bulls"),
    ("LAW", "Lawson", "Racing Bulls"),
    ("STR", "Stroll", "Aston Martin"),
    ("ALO", "Alonso", "Aston Martin"),
    ("OCO", "Ocon", "Haas"),
    ("BEA", "Bearman", "Haas"),
    ("HUL", "Hulkenberg", "Audi"),
    ("BOR", "Bortoleto", "Audi"),
    ("GAS", "Gasly", "Alpine"),
    ("COL", "Colapinto", "Alpine"),
    ("PER", "Perez", "Cadillac"),
    ("BOT", "Bottas", "Cadillac"),
]

PROJECTED_GRID = {
    "NOR": 1,
    "PIA": 2,
    "RUS": 3,
    "ANT": 4,
    "VER": 5,
    "LEC": 6,
    "HAM": 7,
    "HAD": 8,
    "ALB": 9,
    "SAI": 10,
    "ALO": 11,
    "STR": 12,
    "BEA": 13,
    "OCO": 14,
    "GAS": 15,
    "COL": 16,
    "LAW": 17,
    "LIN": 18,
    "HUL": 19,
    "BOR": 20,
    "PER": 21,
    "BOT": 22,
}


def build_projected_grid(data: pd.DataFrame) -> dict[str, int]:
    scores = []
    for code, _, _ in DRIVER_ROSTER_2026:
        history = data[data["Abbreviation"].eq(code)].sort_values("RaceOrder").tail(3)
        average_grid = (
            float(history["GridPosition"].mean())
            if not history.empty
            else float(PROJECTED_GRID[code])
        )
        scores.append((average_grid, PROJECTED_GRID[code], code))

    return {
        code: position
        for position, (_, _, code) in enumerate(sorted(scores), start=1)
    }


def load_grid_positions(projected_grid: dict[str, int]) -> tuple[dict[str, int], dict]:
    if not GRID_OVERRIDE_PATH.exists():
        return projected_grid.copy(), {
            "grid_source": "projected_grid",
            "projection_method": "average_grid_position_last_3_races",
            "grid_override_file": None,
            "overridden_drivers": [],
        }

    raw_grid = json.loads(GRID_OVERRIDE_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw_grid, dict):
        raise ValueError("qualifying_grid.json must be an object like {\"NOR\": 1, \"PIA\": 2}")

    grid = projected_grid.copy()
    unknown_codes = sorted(set(raw_grid) - set(projected_grid))
    if unknown_codes:
        raise ValueError(f"qualifying_grid.json contains unknown driver codes: {unknown_codes}")

    for code, position in raw_grid.items():
        position = int(position)
        if position < 1:
            raise ValueError(f"Grid position for {code} must be 1 or greater")
        grid[code] = position

    return grid, {
        "grid_source": "qualifying_grid",
        "grid_override_file": GRID_OVERRIDE_PATH.name,
        "overridden_drivers": sorted(raw_grid),
    }


def validate_training_race(year: int, gp: str, race_start, now=None) -> None:
    start = pd.to_datetime(race_start, utc=True)
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.to_datetime(now, utc=True)
    if pd.isna(start) or start >= TARGET_START or start >= current:
        raise ValueError(f"Training requires a completed race before Singapore: {year} {gp}")
    if year == 2026 and gp == "Singapore":
        raise ValueError("The target race cannot be a training input")


def load_results(*, offline=False) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    if offline:
        fastf1.Cache.offline_mode(True)
    rows = []

    for race_order, (year, gp) in enumerate(RACES_TO_LOAD):
        try:
            session = fastf1.get_session(year, gp, "R")
            validate_training_race(year, gp, session.date)
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            result = session.results[
                ["Abbreviation", "TeamName", "GridPosition", "Position", "Points"]
            ].copy()
            if result.empty:
                raise RuntimeError("FastF1 returned no classified results")
            result["Year"] = year
            result["GrandPrix"] = gp
            result["RaceOrder"] = race_order
            result["IsStreetCircuit"] = int(gp in STREET_CIRCUITS)
            result["Winner"] = (pd.to_numeric(result["Position"], errors="coerce") == 1).astype(int)
            rows.append(result)
            print(f"Loaded {year} {gp}")
        except Exception as exc:
            print(f"Skipped {year} {gp}: {exc}")

    if not rows:
        raise RuntimeError("No race data loaded. Check FastF1/cache/network availability.")

    data = pd.concat(rows, ignore_index=True)
    loaded_races = set(zip(data["Year"], data["GrandPrix"]))
    required_races = {(year, "Singapore") for year in (2022, 2023, 2024, 2025)} | {(2026, "Azerbaijan")}
    if missing := required_races - loaded_races:
        raise RuntimeError(f"Required Singapore history or latest results are missing: {sorted(missing)}")
    for column, fallback in {"GridPosition": 10, "Position": 20, "Points": 0}.items():
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(fallback)
    data["DriverCode"] = data["Abbreviation"]
    return data


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sort_values(["RaceOrder", "Position"])
    grouped_driver = data.groupby("Abbreviation", group_keys=False)

    data["AvgPoints5"] = grouped_driver["Points"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    data["AvgGrid5"] = grouped_driver["GridPosition"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    data["AvgGrid3"] = grouped_driver["GridPosition"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    data["AvgFinish5"] = grouped_driver["Position"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    data["WinRate10"] = grouped_driver["Winner"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )
    team_race = (
        data.groupby(["RaceOrder", "TeamName"], as_index=False)["Points"]
        .sum()
        .sort_values("RaceOrder")
    )
    team_race["TeamAvgPoints5"] = team_race.groupby("TeamName")["Points"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    data = data.merge(
        team_race[["RaceOrder", "TeamName", "TeamAvgPoints5"]],
        on=["RaceOrder", "TeamName"],
        how="left",
    )

    data["SingaporeExperience"] = data.groupby("Abbreviation")["GrandPrix"].transform(
        lambda x: x.eq("Singapore").shift(fill_value=False).cumsum()
    )
    data["SingaporeWinRate"] = 0.0
    for _, index in data.groupby("Abbreviation").groups.items():
        driver_rows = data.loc[index].sort_values("RaceOrder")
        prior_singapore_wins = []
        wins = []
        for _, row in driver_rows.iterrows():
            prior_singapore_wins.append(float(np.mean(wins)) if wins else 0.0)
            if row["GrandPrix"] == "Singapore":
                wins.append(row["Winner"])
        data.loc[driver_rows.index, "SingaporeWinRate"] = prior_singapore_wins

    data["AvgPoints5"] = data["AvgPoints5"].fillna(0)
    data["AvgGrid5"] = data["AvgGrid5"].fillna(data["GridPosition"])
    data["AvgGrid3"] = data["AvgGrid3"].fillna(data["GridPosition"])
    data["AvgFinish5"] = data["AvgFinish5"].fillna(14)
    data["WinRate10"] = data["WinRate10"].fillna(0)
    data["TeamAvgPoints5"] = data["TeamAvgPoints5"].fillna(0)
    return data


def build_model() -> Pipeline:
    base = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=220,
        max_leaf_nodes=15,
        l2_regularization=0.1,
        random_state=42,
    )
    calibrated = CalibratedClassifierCV(
        base,
        method="sigmoid",
        cv=StratifiedGroupKFold(n_splits=3),
    )
    return Pipeline(
        [
            (
                "encode",
                ColumnTransformer(
                    [
                        (
                            "categorical",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                            CATEGORICAL_FEATURES,
                        ),
                        ("numeric", "passthrough", NUMERIC_FEATURES),
                    ],
                    remainder="drop",
                ),
            ),
            ("model", calibrated),
        ]
    )


def build_prediction_rows(data: pd.DataFrame, grid_positions: dict[str, int]) -> pd.DataFrame:
    rows = []

    for code, driver, team in DRIVER_ROSTER_2026:
        history = data[data["Abbreviation"].eq(code)].sort_values("RaceOrder")
        team_history = (
            data[data["TeamName"].eq(team)]
            .groupby("RaceOrder", as_index=False)["Points"]
            .sum()
            .sort_values("RaceOrder")
        )
        recent_five = history.tail(5)
        recent_three = history.tail(3)
        recent_ten = history.tail(10)
        singapore_history = history[history["GrandPrix"].eq("Singapore")]
        rows.append(
            {
                "Abbreviation": code,
                "driver": driver,
                "TeamName": team,
                "GridPosition": grid_positions[code],
                "DriverCode": code,
                "AvgPoints5": float(recent_five["Points"].mean()) if not history.empty else 0.0,
                "AvgGrid3": float(recent_three["GridPosition"].mean()) if not history.empty else grid_positions[code],
                "AvgGrid5": float(recent_five["GridPosition"].mean()) if not history.empty else grid_positions[code],
                "AvgFinish5": float(recent_five["Position"].mean()) if not history.empty else 14.0,
                "WinRate10": float(recent_ten["Winner"].mean()) if not history.empty else 0.0,
                "TeamAvgPoints5": float(team_history.tail(5)["Points"].mean()) if not team_history.empty else 0.0,
                "SingaporeExperience": float(len(singapore_history)),
                "SingaporeWinRate": float(singapore_history["Winner"].mean()) if not singapore_history.empty else 0.0,
                "IsStreetCircuit": 1,
            }
        )

    return pd.DataFrame(rows)


def calculate_form_prior(pred: pd.DataFrame) -> pd.Series:
    avg_points_max = pred["AvgPoints5"].clip(lower=0).max()
    team_points_max = pred["TeamAvgPoints5"].clip(lower=0).max()
    form_score = (
        (1 / pred["GridPosition"].clip(lower=1))
        + pred["AvgPoints5"].clip(lower=0) / (avg_points_max if avg_points_max else 1)
        + pred["TeamAvgPoints5"].clip(lower=0) / (team_points_max if team_points_max else 1)
        + pred["WinRate10"].clip(lower=0)
        + pred["SingaporeWinRate"].clip(lower=0)
    )
    return form_score / form_score.sum()


def blend_probabilities(
    model_probs: np.ndarray,
    form_prior: pd.Series,
    model_weight: float,
    floor: float,
) -> pd.Series:
    blended_score = (
        (model_weight * model_probs)
        + ((1 - model_weight) * form_prior.to_numpy())
        + floor
    )
    return pd.Series(blended_score / blended_score.sum(), index=form_prior.index)


def apply_probability_postprocess(
    pred: pd.DataFrame,
    model_probs: np.ndarray,
    model_weight: float,
    floor: float,
) -> pd.Series:
    form_prior = calculate_form_prior(pred)
    return blend_probabilities(model_probs, form_prior, model_weight, floor)


def summarize_backtest(results: list[dict], probability_rows: list[dict]) -> dict:
    if not results:
        return {
            "races_tested": 0,
            "top1_accuracy": None,
            "top3_accuracy": None,
            "top5_accuracy": None,
            "mean_winner_rank": None,
            "log_loss": None,
            "brier_score": None,
        }

    actual = [row["actual"] for row in probability_rows]
    probability = [row["probability"] for row in probability_rows]
    return {
        "races_tested": len(results),
        "top1_accuracy": round(float(np.mean([row["winner_rank"] == 1 for row in results])), 4),
        "top3_accuracy": round(float(np.mean([row["winner_rank"] <= 3 for row in results])), 4),
        "top5_accuracy": round(float(np.mean([row["winner_rank"] <= 5 for row in results])), 4),
        "mean_winner_rank": round(float(np.mean([row["winner_rank"] for row in results])), 2),
        "log_loss": round(float(log_loss(actual, probability)), 4),
        "brier_score": round(float(brier_score_loss(actual, probability)), 4),
    }


def rank_backtest_result(item: dict) -> tuple:
    summary = item["summary"]
    return (
        summary["top1_accuracy"] or 0,
        -1 * (summary["mean_winner_rank"] or 999),
        -1 * (summary["log_loss"] or 999),
        -1 * (summary["brier_score"] or 999),
    )


def tune_walk_forward_postprocess(data: pd.DataFrame, min_training_races: int = 8) -> dict:
    race_keys = (
        data[["RaceOrder", "Year", "GrandPrix"]]
        .drop_duplicates()
        .sort_values("RaceOrder")
        .to_dict("records")
    )
    candidates = {
        index: {"config": config, "races": [], "probability_rows": []}
        for index, config in enumerate(POSTPROCESS_CANDIDATES)
    }

    for race in race_keys:
        prior_race_count = sum(item["RaceOrder"] < race["RaceOrder"] for item in race_keys)
        if prior_race_count < min_training_races:
            continue

        train = data[data["RaceOrder"] < race["RaceOrder"]]
        target = data[data["RaceOrder"] == race["RaceOrder"]].copy()
        if train["Winner"].nunique() < 2 or target.empty:
            continue

        model = build_model()
        model.fit(train[FEATURES], train["Winner"], groups=train["RaceOrder"])
        model_probs = model.predict_proba(target[FEATURES])[:, 1]
        form_prior = calculate_form_prior(target)

        for candidate in candidates.values():
            config = candidate["config"]
            target["probability"] = blend_probabilities(
                model_probs,
                form_prior,
                config["model_weight"],
                config["floor"],
            )
            ranked = target.sort_values("probability", ascending=False).reset_index(drop=True)
            winner_index = ranked.index[ranked["Winner"].eq(1)]
            if len(winner_index) == 0:
                continue

            winner_rank = int(winner_index[0] + 1)
            winner = ranked.loc[winner_index[0]]
            candidate["races"].append(
                {
                    "race": f"{race['Year']} {race['GrandPrix']}",
                    "winner": str(winner["Abbreviation"]),
                    "winner_rank": winner_rank,
                    "winner_probability": round(float(winner["probability"]), 4),
                }
            )
            candidate["probability_rows"].extend(
                {
                    "actual": int(row["Winner"]),
                    "probability": float(row["probability"]),
                }
                for _, row in ranked.iterrows()
            )

    evaluated = []
    for candidate in candidates.values():
        evaluated.append(
            {
                "config": candidate["config"],
                "races": candidate["races"],
                "summary": summarize_backtest(candidate["races"], candidate["probability_rows"]),
            }
        )

    best = max(evaluated, key=rank_backtest_result)
    return {
        "selected_config": best["config"],
        "races": best["races"],
        "summary": best["summary"],
        "candidates_tested": len(evaluated),
        "selection_metric": "top1_accuracy, then mean_winner_rank, log_loss, brier_score",
    }


def run_walk_forward_backtest(
    data: pd.DataFrame,
    model_weight: float,
    floor: float,
    evaluation_start_year: int,
    min_training_races: int = 8,
) -> dict:
    race_keys = (
        data[["RaceOrder", "Year", "GrandPrix"]]
        .drop_duplicates()
        .sort_values("RaceOrder")
        .to_dict("records")
    )
    results = []
    probability_rows = []

    for race in race_keys:
        if race["Year"] < evaluation_start_year:
            continue
        prior_race_count = sum(item["RaceOrder"] < race["RaceOrder"] for item in race_keys)
        if prior_race_count < min_training_races:
            continue

        train = data[data["RaceOrder"] < race["RaceOrder"]]
        target = data[data["RaceOrder"] == race["RaceOrder"]].copy()
        if train["Winner"].nunique() < 2 or target.empty:
            continue

        model = build_model()
        model.fit(train[FEATURES], train["Winner"], groups=train["RaceOrder"])
        model_probs = model.predict_proba(target[FEATURES])[:, 1]
        target["probability"] = apply_probability_postprocess(
            target,
            model_probs,
            model_weight,
            floor,
        )
        ranked = target.sort_values("probability", ascending=False).reset_index(drop=True)
        winner_index = ranked.index[ranked["Winner"].eq(1)]
        if len(winner_index) == 0:
            continue

        winner_rank = int(winner_index[0] + 1)
        winner = ranked.loc[winner_index[0]]
        results.append(
            {
                "race": f"{race['Year']} {race['GrandPrix']}",
                "winner": str(winner["Abbreviation"]),
                "winner_rank": winner_rank,
                "winner_probability": round(float(winner["probability"]), 4),
            }
        )
        probability_rows.extend(
            {
                "actual": int(row["Winner"]),
                "probability": float(row["probability"]),
            }
            for _, row in ranked.iterrows()
        )

    return {
        "races": results,
        "summary": summarize_backtest(results, probability_rows),
    }


def export_inference(pred, predictions, metadata):
    """Freeze pre-race inputs so qualifying updates change only GridPosition."""
    base = json.dumps([predictions, metadata], sort_keys=True, separators=(",", ":"), allow_nan=False)
    bundle = {
        "schema_version": 1,
        "base_hash": hashlib.sha256(base.encode()).hexdigest(),
        "model_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "sklearn_version": sklearn.__version__,
        "features": FEATURES,
        "circuit_win_rate_feature": "SingaporeWinRate",
        "rows": pred[["driver", *FEATURES]].to_dict(orient="records"),
    }
    (OUTPUT_DIR / "singapore_inference.json").write_text(json.dumps(bundle, indent=2, allow_nan=False), encoding="utf-8")


def export_existing_inference():
    """Reconstruct from cached training inputs without retraining or replacing forecasts."""
    data = engineer_features(load_results(offline=True))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    predictions = json.loads(PREDICTIONS_PATH.read_text(encoding="utf-8"))
    if metadata["prediction_input"]["grid_source"] != "projected_grid":
        raise ValueError("Existing export requires the original pre-qualifying forecast")
    if data[["Year", "GrandPrix"]].drop_duplicates().shape[0] != metadata["training_races_loaded"]:
        raise ValueError("Cached training races differ from the original training run")
    pred = build_prediction_rows(data, build_projected_grid(data))
    model = joblib.load(MODEL_PATH)
    config = metadata["prediction_postprocess"]
    pred["probability"] = apply_probability_postprocess(pred, model.predict_proba(pred[FEATURES])[:, 1],
                                                       config["model_weight"], config["floor_before_normalization"])
    reproduced = [{"driver": row["driver"], "team": row["TeamName"], "probability": round(float(row["probability"]), 4)}
                  for _, row in pred.sort_values("probability", ascending=False).iterrows()]
    if reproduced != predictions:
        raise ValueError("Frozen inputs do not reproduce the published forecast")
    export_inference(pred, predictions, metadata)
    print("Exported frozen inference features; all published probabilities reproduced exactly.")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = engineer_features(load_results())
    projected_grid = build_projected_grid(data)
    grid_positions, grid_metadata = load_grid_positions(projected_grid)
    tuning_data = data[data["Year"] < 2026].copy()
    tuned_postprocess = tune_walk_forward_postprocess(tuning_data)
    model_weight = tuned_postprocess["selected_config"]["model_weight"]
    floor = tuned_postprocess["selected_config"]["floor"]
    model = build_model()
    x = data[FEATURES]

    model.fit(x, data["Winner"], groups=data["RaceOrder"])
    pred = build_prediction_rows(data, grid_positions)
    model_probs = model.predict_proba(pred[FEATURES])[:, 1]
    pred["probability"] = apply_probability_postprocess(pred, model_probs, model_weight, floor)
    backtest = run_walk_forward_backtest(
        data,
        model_weight,
        floor,
        evaluation_start_year=2026,
    )

    predictions = [
        {
            "driver": row["driver"],
            "team": row["TeamName"],
            "probability": round(float(row["probability"]), 4),
        }
        for _, row in pred.sort_values("probability", ascending=False).iterrows()
    ]
    latest_race = data.sort_values("RaceOrder").iloc[-1]
    earliest_race = data.sort_values("RaceOrder").iloc[0]

    metadata = {
        "race": "Singapore GP",
        "race_date": "2026-10-11",
        "circuit": "Marina Bay Street Circuit",
        "model_version": "singapore-hgb-calibrated-1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_cutoff": {
            "year": int(latest_race["Year"]),
            "race": str(latest_race["GrandPrix"]),
        },
        "training_period": {
            "start": f"{int(earliest_race['Year'])} {earliest_race['GrandPrix']}",
            "end": f"{int(latest_race['Year'])} {latest_race['GrandPrix']}",
        },
        "driver_roster": {
            "status": "projected_from_latest_lineup",
            "basis": "Latest 2026 lineup; Singapore entries are not yet confirmed",
            "source": "https://www.formula1.com/en/results/2026/races/1295/azerbaijann/race-result",
        },
        "training_samples": int(len(data)),
        "training_races_loaded": int(data[["Year", "GrandPrix"]].drop_duplicates().shape[0]),
        "features": FEATURES,
        "validation": {
            "method": "walk_forward_backtest",
            **backtest["summary"],
        },
        "prediction_postprocess": {
            "model_weight": model_weight,
            "form_prior_weight": round(1 - model_weight, 4),
            "floor_before_normalization": floor,
            "selection": {
                "method": "walk_forward_grid_search",
                "candidates_tested": tuned_postprocess["candidates_tested"],
                "selection_metric": tuned_postprocess["selection_metric"],
                "tuning_period": "through 2025",
                "tuning_summary": tuned_postprocess["summary"],
            },
        },
        "prediction_input": grid_metadata,
        "backtest": backtest,
    }

    joblib.dump(model, MODEL_PATH)
    PREDICTIONS_PATH.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    export_inference(pred, predictions, metadata)

    print(json.dumps(metadata, indent=2))
    for i, item in enumerate(predictions, 1):
        print(f"P{i}: {item['driver']:12} {item['team']:16} {item['probability'] * 100:5.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-inference-only", action="store_true")
    args = parser.parse_args()
    export_existing_inference() if args.export_inference_only else main()
