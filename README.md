# An extended deep energy method for thermo-mechanical crack propagation

Reference implementation for the paper of the same name. The crack is a sharp polyline
throughout. The temperature and the displacement are represented by two networks that receive
the crack through a scalar embedding function, discontinuous across the crack and smooth
elsewhere, so both fields can jump across the crack without a regularization length, and the
displacement is enriched near the tip by the Williams expansion with trainable amplitudes. The
temperature is obtained by minimizing an incremental functional of transient heat conduction and
the displacement by minimizing the thermoelastic potential energy, in a staggered sequence at
every load step, on Monte Carlo integration points that are stratified over square background
elements, densified near the tip and redrawn during training. The stress intensity factors are
extracted by the interaction integral with the thermal area term of Wilson and Yu, each
extraction is checked by a sweep of the contour radius, and the crack advances at the maximum
hoop stress angle when the energy release rate of the kink reaches the critical value at the
crack-tip temperature.

Preprint: https://arxiv.org/abs/2610.09433

Contact: han.zhang7@unsw.edu.au

## Layout

    tmdem/         the solver: crack embedding, networks and tip enrichment, integration
                   points, thermal and mechanical energies, training loops, and the
                   interaction integral with the thermal area term
    examples/      one script per numerical example of the paper; each runs on its own
    fem/           the finite element solvers of the reference solutions

## Requirements

Python 3.10 with the packages in `requirements.txt`. A CUDA device is expected for the
propagating examples; the code also runs on the CPU, slowly. The finite element references need
`scikit-fem`, and `fem/pf_fem.py` uses `pypardiso` for the direct solves if it is
installed.

## Running the examples

Every script writes to `runs/<case>/` below the folder it is started from, or to the folder
given by `--out`, and `--resume` continues an interrupted run from its latest checkpoint. The
defaults of each script are the settings of the runs reported in the paper.

    python examples/stationary_crack.py --nx 48     # stationary thermal edge crack,
    python examples/stationary_crack.py --nx 72     #   the three grids of the paper
    python examples/stationary_crack.py --nx 96

    python examples/sen_tension.py --dT 2.5         # tension, heated to +50 K
    python examples/sen_tension.py --dT -2.5        #   cooled to -50 K
    python examples/sen_tension.py --dT 0           #   isothermal

    python examples/sen_shear_fgm.py --dT 30        # graded shear, heated by 30 K
    python examples/sen_shear_fgm.py --dT -30       #   cooled by 30 K
    python examples/sen_shear_fgm.py --dT 0         #   isothermal

    python examples/cruciform.py --load-type I      # notched cruciform, mechanical load
    python examples/cruciform.py --load-type II     #   thermal load, top face free
    python examples/cruciform.py --load-type III    #   both

In the tension test `--dT` is the temperature change per load step, applied over the first 20
of 106 steps; in the shear test it is the total change, applied over the first 12 of 141 steps.
The cruciform repeats of the paper use `--seed 2025`, `--seed 1` and `--seed 2`. Each load step
writes its fields, the extracted stress intensity factors, the radius sweep and a checkpoint to
`results/step_<n>/`, and the load history and the propagation log to `results/`.

The seed sets the network initialization. The integration points are drawn by a Latin hypercube
sampler that is not seeded, so two runs with the same seed differ in their integration points;
the paper reports this run-to-run scatter for every example.

For orientation, the stationary crack takes about 17 minutes on a laptop GPU at its finest
grid, a tension case 8 to 9 hours and a shear case 7 to 12 hours on one V100, and a cruciform
case 3.5 to 4 hours on one H200.

## Finite element references

The phase-field reference of the tension and shear tests is a staggered finite element solver
(AT2, hybrid split, bilinear quadrilaterals) with transient conduction on the degraded
conductivity. The graded shear curves of the paper were computed with

    python fem/pf_fem.py --example shear --split hybrid --fgm --dT_final 30 \
        --alpha_mode ps3d --heat_mass lumped --l 0.01 --refine 9 \
        --delta_max 3.5e-3 --nsteps 700 --stag_max 100 --out runs_fem/shear_fgm_dT30

and likewise with `--dT_final -30` and `0`; `--example tension` selects the tension test.

The sharp-crack reference of the graded shear test holds the crack at its initial length and
obtains the stress intensity factors from the energy released by a straight extension and the
ratio of the face openings. `sharp_reference_thermal.py` runs its three thermal cases to
initiation, `sharp_reference.py` is its isothermal check, and `wang_sharp_fe.py` applies it to
the stationary thermal edge crack:

    python fem/sharp_reference_thermal.py
    python fem/sharp_reference.py
    python fem/wang_sharp_fe.py

## Citation

If you use this code in work you publish, please cite the paper:

    @article{zhang2026extended,
      author  = {Zhang, Han and Makki Alamdari, Mehrisadat and Shahbodagh, Babak
                 and Vahab, Mohammad and Anitescu, Cosmin and Rabczuk, Timon
                 and Atroshchenko, Elena},
      title   = {An extended deep energy method for thermo-mechanical crack
                 propagation},
      journal = {arXiv preprint arXiv:2610.09433},
      year    = {2026}
    }

## License

MIT, see `LICENSE`.
