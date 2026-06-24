import numpy as np

from bolr.model.priors import BlockDynamicsSpec, BlockPriorSpec, assemble_block_prior, assemble_block_process_noise
from bolr.model.state_layout import make_state_layout


def test_block_prior_and_process_noise_assembly() -> None:
    layout = make_state_layout([{"name": "surface", "shape": (2,)}, {"name": "history", "shape": (1,)}])
    posterior = assemble_block_prior(
        layout,
        [
            BlockPriorSpec("surface", isotropic_scale=2.0),
            BlockPriorSpec("history", diagonal=np.array([0.5])),
        ],
    )
    assert np.allclose(np.diag(posterior.covariance), [4.0, 4.0, 0.5])
    process = assemble_block_process_noise(
        layout,
        [
            BlockDynamicsSpec("surface", isotropic_process_variance=0.2),
            BlockDynamicsSpec("history", frozen=True),
        ],
    )
    assert np.allclose(np.diag(process), [0.2, 0.2, 0.0])

