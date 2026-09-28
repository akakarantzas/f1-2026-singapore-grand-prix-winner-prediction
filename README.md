# F1 2026 Singapore Grand Prix Winner Prediction

Standalone machine learning project for the 2026 Singapore Grand Prix at Marina
Bay Street Circuit on October 11, 2026. [Official race information](https://www.formula1.com/en/racing/2026/singapore).

This repository owns data loading, feature engineering, training, validation,
and forecast exports. ChicaneAI serves the exported JSON without training or
loading the pickle model during an API request.

## Approach

- FastF1 race classifications from 2022 Singapore and the 2023–2026 seasons,
  with the initial forecast's data cutoff at the completed 2026 Azerbaijan GP.
- Rolling driver and team form, projected starting positions, Singapore
  experience and win rate, and street-circuit context.
- Histogram gradient boosting with sigmoid calibration and race-grouped folds.
- Walk-forward tuning through 2025, followed by separate 2026 backtesting.
- A model/form blend normalized across the 22-driver forecast grid.

Training excludes the target race and races that have not started. Required
Singapore history and the latest Azerbaijan results must load before publication.
Historical measurements evaluate completed races, not the upcoming Singapore GP.

The entry list is projected from the latest 2026 lineup. The forecast does not
claim a confirmed Singapore entry list or qualifying grid. Update the roster if
official entries change and regenerate after qualifying.

## Run

```bash
pip install -r requirements.txt
python train_singapore.py
python -m unittest discover -s tests -v
```

Use `requirements.lock.txt` for the exact dependency set used by Docker.
FastF1 stores downloaded results in the ignored `cache/` directory.

```bash
docker compose up --build
docker compose run --rm --build train python -m unittest discover -s tests -v
```

Docker uses Python 3.12 and exports artifacts into `artifacts/`. Copy validated
exports to the repository root when publishing a new forecast. The root exports
are the published snapshot; tests validate them along with feature behavior.

## Qualifying grid

Before qualifying, projected positions use each driver's last three recorded
grids. After qualifying, copy `qualifying_grid.example.json` to
`qualifying_grid.json`, supply official positions, and retrain.

## Outputs

- `singapore_predictions.json`: ranked driver win probabilities.
- `singapore_metadata.json`: race, data cutoff, features and measured backtests.
- `singapore_model.pkl`: fitted model for reproducibility.

The September 28, 2026 export uses 1,748 driver results from 86 races, through
the 2026 Azerbaijan GP. The 2026 walk-forward check covers 15 races: the winner
ranked first in 9 and within the top three in 13. These are historical checks
across multiple circuits, not measured Singapore forecast accuracy.

## ChicaneAI integration

Run the app's sync script after training and publishing the root artifacts:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/sync_azerbaijan_model.ps1 -SourceRepo .tmp-singapore-model-repo -PredictionName singapore -IncludeModel
```

The sync archives the forecast before copying its files into
`backend/app/models/`. The Predictions tab and homepage consume the API's
Singapore response; past Azerbaijan forecasts remain in History.

## License

[MIT](LICENSE).
