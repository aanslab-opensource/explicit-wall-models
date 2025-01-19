# Explicit Formulations of Widely-Used Wall Models for Large-Eddy Simulation ???

[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14423351.svg)](https://doi.org/10.5281/zenodo.14423351)


This repository contains information and code to reproduce the results presented in the article
```bibtex
@online{expwallmodinc2025,
  title={Explicit Formulations of Widely-Used Wall Models for Large-Eddy Simulation ??},
  author={Nuca, Roberto and Mukha, Timofey and Parsani, Matteo},
  year={2025},
  month={??},
  eprint={2412.17117},
  eprinttype={arxiv},
  eprintclass={math.NA}
}
```

If you find these results useful, please cite the article mentioned above. If you use the implementations provided here, please **also** cite this repository as
```bibtex
@misc{expwallmodinc2025,
  title={Reproducibility repository for "Explicit Formulations of Widely-Used Wall Models for Large-Eddy Simulation ??"},
  author={Nuca, Roberto and Mukha, Timofey and Parsani, Matteo},
  year={2025},
  howpublished={\url{https://github.com/aanslab-opensource/explicit-wall-models}},
  doi={10.5281/zenodo.????}
}
```


## Abstract

Algebraic wall models for large-eddy simulations may suffer from robustness issues when applied in complex flow configurations. On the other hand, models based on ordinarydifferential equations (ODE-based models) are robust, but are slower to converge, computationally more expensive, and are harder to implement. The latter is exacerbated by the plethora of hardware back-ends a modern code is expected to support. Here, approximate explicit formulations for, arguably, the three most widely-used wall models are provided: the Equilibrium ODE model, Spalding's model, and Reichardt's model. Using these explicit versions of the models makes them trivial to implement and the computation of the stress unconditionally stable. The resulting expressions are compact, yet introduce at most a 1\% relative error in the models' velocity profile for all physically realizable values of the von Karman constant and intercept of the logarithmic law. Furthermore, tuned approximations with at most 0.05\% relative error are provided for two sets of logarithmic law parameters: modern values based on high-Reynolds-number data and classical ones used in the original formulation of the wall models. In addition to using explicit models directly, they can be utilized to provide an initial guess for the iterative methods used to solve the respective original counterparts. We demonstrate that this leads to increased robustness and, for the ODE model, to faster convergence. While our work covers three wall models, the provided approximation approach is expected to be applicable to any model based on a one-dimensional coupling of the linear and logarithmic laws of the wall.


## Numerical experiments

The numerical experiments use [Python](https://www.python.org) and [OpenFOAM-v23.06](https://develop.openfoam.com/Development/openfoam/-/tree/OpenFOAM-v2306?ref_type=tags).

To run the Python code, you need Python 3, and the packages `jupyter`, `numpy`, `scipy`, and `matplotlib`.
The Python code has been tested with the following versions, but other versions may work:

    - Python 3.13
    - Numpy 2.1.3
    - SciPy 1.14.1
    - Matplotlib 3.9.2
    - sklearn

The results can be reproduced by running  .....


## Disclaimer

Everything is provided as is and without warranty. Use at your own risk!
