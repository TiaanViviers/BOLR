import pandas as pd
import pytest

from bolr.config.foundation import HistoricalDatasetConfig
from bolr.data.candidate_grid import CandidateGrid
from bolr.data.historical_dataset import HistoricalDataset


def test_historical_dataset_blocks_outcome_reveal_before_prediction() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2021-01-29", "2021-01-29", "2021-01-30", "2021-01-30"],
            "config_id": [0, 1, 0, 1],
            "entry_percentage": [0.1, 0.2, 0.1, 0.2],
            "sl_trail_percentage": [0.3, 0.4, 0.3, 0.4],
            "pnl": [1.0, 2.0, 1.5, 2.5],
        }
    )
    grid = CandidateGrid(
        config_ids=pd.Series([0, 1]).to_numpy(),
        entry_values=pd.Series([0.1, 0.2]).to_numpy(),
        stop_values=pd.Series([0.3, 0.4]).to_numpy(),
        pair_to_id={(0.1, 0.3): 0, (0.2, 0.4): 1},
        grid_shape=(1, 2),
    )
    dataset = HistoricalDataset(
        frame,
        candidate_grid=grid,
        config=HistoricalDatasetConfig(
            expected_rows=4,
            expected_columns=5,
            expected_dates=2,
            expected_rows_per_date=2,
            expected_start_date="2021-01-29",
            expected_end_date="2021-01-30",
        ),
    )
    with pytest.raises(RuntimeError):
        dataset.reveal_outcomes("2021-01-29")
    dataset.get_predictors("2021-01-29")
    outcomes = dataset.reveal_outcomes("2021-01-29")
    assert outcomes.pnl.tolist() == [1.0, 2.0]

