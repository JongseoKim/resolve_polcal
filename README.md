Bayesian polarization calibration and imaging pipeline using resolve software
==============================================================================

This repository contains the code in the publication [Bayesian polarization calibration and imaging in very long baseline interferometry](https://arxiv.org/abs/2511.16556).

Installation
------------

Install dependencies: [nifty8](https://gitlab.mpcdf.mpg.de/ift/nifty), [ducc](https://gitlab.mpcdf.mpg.de/mtr/ducc), [resolve](https://gitlab.mpcdf.mpg.de/ift/resolve), numpy, astropy

Optional dependencies: mpi4py, configparser, python-casacore, h5py, matplotlib

Instruction
------------

Download the example data (MOJAVE VLBA 3C 273 at 15 GHz):
https://zenodo.org/records/17699222

Run the example code with parallelization:

mpiexec -n 4 python3 VLBI_pol_imaging.py conf_polcal.cfg



Licensing terms
---------------

This package is licensed under the terms of the
[GPLv3](https://www.gnu.org/licenses/gpl.html) and is distributed
*without any warranty*.


