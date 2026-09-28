import json
import unittest
from pathlib import Path

import pandas as pd

from train_singapore import (
    FEATURES,
    PROJECTED_GRID,
    build_model,
    build_prediction_rows,
    build_projected_grid,
    engineer_features,
    validate_training_race,
)


def result_row(race_order, driver, team, points, grid, finish, race="Test"):
    return {
        "RaceOrder": race_order,
        "Year": 2025,
        "GrandPrix": race,
        "Abbreviation": driver,
        "DriverCode": driver,
        "TeamName": team,
        "Points": points,
        "GridPosition": grid,
        "Position": finish,
        "Winner": int(finish == 1),
        "IsStreetCircuit": 0,
    }


class FeatureEngineeringTests(unittest.TestCase):
    def test_singapore_features_use_only_prior_singapore_results(self):
        data = pd.DataFrame([
            result_row(0, "NOR", "McLaren", 25, 1, 1, race="Singapore"),
            result_row(1, "NOR", "McLaren", 25, 1, 1, race="Azerbaijan"),
            result_row(2, "NOR", "McLaren", 18, 2, 2, race="Singapore"),
        ])
        engineered = engineer_features(data)
        self.assertEqual(engineered["SingaporeExperience"].tolist(), [0, 1, 1])
        self.assertEqual(engineered["SingaporeWinRate"].tolist(), [0.0, 1.0, 1.0])
        prediction = build_prediction_rows(engineered, PROJECTED_GRID)
        norris = prediction[prediction["Abbreviation"].eq("NOR")].iloc[0]
        self.assertEqual(norris["SingaporeExperience"], 2)
        self.assertEqual(norris["SingaporeWinRate"], 0.5)

    def test_training_excludes_target_and_future_races(self):
        validate_training_race(2026, "Azerbaijan", "2026-09-26T11:00:00Z", now="2026-09-28T12:00:00Z")
        for gp, start in [("Singapore", "2026-10-11T12:00:00Z"), ("Bahrain", "2026-10-04T07:00:00Z"), ("United States", "2026-10-25T12:00:00Z")]:
            with self.subTest(gp=gp), self.assertRaises(ValueError):
                validate_training_race(2026, gp, start, now="2026-09-28T12:00:00Z")

    def test_team_form_uses_prior_complete_races(self):
        data = pd.DataFrame(
            [
                result_row(0, "NOR", "McLaren", 25, 1, 1),
                result_row(0, "PIA", "McLaren", 18, 2, 2),
                result_row(1, "NOR", "McLaren", 15, 2, 3),
                result_row(1, "PIA", "McLaren", 25, 1, 1),
            ]
        )

        engineered = engineer_features(data)
        second_race = engineered[engineered["RaceOrder"].eq(1)]

        self.assertEqual(second_race["TeamAvgPoints5"].nunique(), 1)
        self.assertEqual(float(second_race["TeamAvgPoints5"].iloc[0]), 43.0)

    def test_prediction_form_includes_latest_race(self):
        data = pd.DataFrame(
            [
                result_row(index, "NOR", "McLaren", points, index + 1, index + 1)
                for index, points in enumerate([1, 2, 3, 4, 25])
            ]
        )
        engineered = engineer_features(data)

        predictions = build_prediction_rows(engineered, PROJECTED_GRID)
        norris = predictions[predictions["Abbreviation"].eq("NOR")].iloc[0]

        self.assertEqual(float(norris["AvgPoints5"]), 7.0)

    def test_calibration_folds_keep_races_together(self):
        model = build_model()
        splitter = model.named_steps["model"].cv
        groups = pd.Series([race for race in range(6) for _ in range(4)])
        target = pd.Series([1, 0, 0, 0] * 6)
        features = pd.DataFrame(0, index=range(len(target)), columns=FEATURES)

        for train_index, test_index in splitter.split(features, target, groups):
            train_races = set(groups.iloc[train_index])
            test_races = set(groups.iloc[test_index])
            self.assertTrue(train_races.isdisjoint(test_races))

    def test_projected_grid_uses_recent_qualifying_form(self):
        data = pd.DataFrame(
            [
                result_row(0, "NOR", "McLaren", 10, 10, 5),
                result_row(1, "NOR", "McLaren", 10, 1, 5),
                result_row(2, "NOR", "McLaren", 10, 1, 5),
                result_row(0, "PIA", "McLaren", 10, 2, 5),
                result_row(1, "PIA", "McLaren", 10, 2, 5),
                result_row(2, "PIA", "McLaren", 10, 2, 5),
            ]
        )

        projected = build_projected_grid(data)

        self.assertLess(projected["PIA"], projected["NOR"])


class ArtifactTests(unittest.TestCase):
    def test_exported_probabilities_are_complete_and_normalized(self):
        path = Path(__file__).resolve().parents[1] / "singapore_predictions.json"
        predictions = json.loads(path.read_text(encoding="utf-8"))
        drivers = [item["driver"] for item in predictions]

        self.assertEqual(len(predictions), 22)
        self.assertEqual(len(set(drivers)), 22)
        self.assertAlmostEqual(sum(item["probability"] for item in predictions), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
