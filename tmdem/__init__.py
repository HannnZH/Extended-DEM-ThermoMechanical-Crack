"""
tmdem: an extended deep energy method for thermo-mechanical crack propagation.

The crack is a sharp polyline throughout.  The temperature and the displacement are
represented by two networks that receive the crack through a scalar embedding function,
discontinuous across the crack and smooth elsewhere, and the displacement is enriched near the
tip by the Williams expansion with trainable amplitudes.  Each load step minimizes an
incremental functional of transient heat conduction and then the thermoelastic potential
energy on Monte Carlo integration points, extracts the stress intensity factors with the
interaction integral and its thermal area term, and advances the crack at the maximum hoop
stress angle when the energy release rate of the kink reaches the critical value.

Four modules:

    crack_embedding   the crack polyline and the embedding function Gamma(x, y)
    solver            networks, tip enrichment, integration points, the thermal and the
                      mechanical energies and the training loops
    sif_estimation    interaction integral with the thermal area term, radius sweep,
                      maximum hoop stress direction and kinked energy release rate
    utils             checkpoints, plotting grids and logging for the example scripts

Runs are launched from the scripts in examples/; see README.md.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""

__version__ = "1.0.0"
