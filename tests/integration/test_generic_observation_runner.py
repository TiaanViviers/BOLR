import numpy as np
import pandas as pd

from bolr.config.foundation import CrossGroupLogisticConfig, HistoricalDatasetConfig, HistoricalRunConfig, OrderedPartitionConfig
from bolr.data.candidate_grid import CandidateGrid
from bolr.data.historical_dataset import HistoricalDataset
from bolr.evaluation.prequential_runner import run_historical_replay
from bolr.observations.cross_group_logistic import CrossGroupLogisticObservationModel
from bolr.targets.ordered_partition import OrderedPartitionBuilder


def test_generic_runner_accepts_candidate_b_components(tmp_path) -> None:
    dates = ["2021-01-29", "2021-01-30", "2021-01-31", "2021-02-01"]
    frame = pd.DataFrame(
        {
            "date": sum(([d, d] for d in dates), []),
            "config_id": [0, 1] * len(dates),
            "entry_percentage": [0.1, 0.2] * len(dates),
            "sl_trail_percentage": [0.3, 0.4] * len(dates),
            "pnl": [2.0, -1.0, 1.5, -0.5, 1.2, 0.8, 2.1, -0.7],
        }
    )
    grid = CandidateGrid(
        config_ids=np.array([0, 1]),
        entry_values=np.array([0.1, 0.2]),
        stop_values=np.array([0.3, 0.4]),
        pair_to_id={(0.1, 0.3): 0, (0.2, 0.4): 1},
        grid_shape=(1, 2),
    )
    dataset = HistoricalDataset(
        frame,
        candidate_grid=grid,
        config=HistoricalDatasetConfig(
            expected_rows=8,
            expected_columns=5,
            expected_dates=4,
            expected_rows_per_date=2,
            expected_start_date="2021-01-29",
            expected_end_date="2021-02-01",
        ),
    )
    result = run_historical_replay(
        dataset,
        np.array([[1.0], [0.5]]),
        target_builder=OrderedPartitionBuilder(OrderedPartitionConfig()),
        observation_model=CrossGroupLogisticObservationModel(CrossGroupLogisticConfig()),
        config=HistoricalRunConfig(warm_up_days=2, outputs_dir=str(tmp_path / "outputs")),
        run_label="historical_candidate_b_cross_group",
    )
    assert len(result.predictions) == 2
    assert (result.run_dir / "predictions.parquet").exists()

