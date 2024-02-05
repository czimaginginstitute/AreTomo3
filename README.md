# AreTomo3
A fully automated preprocessing pipeline that enables real time generation of cryoET tomograms with integrated correction of beam induced motion, CTF estimation, tomographic alignment and reconstruction in a single multi-GPU accelerated application.

## Installation
AreTomo3 is developed on Linux platform equipped with at least one Nvidia GPU card. To compile from the source, follow the steps below:

1.      git clone https://github.com/czimaginginstitute/AreTomo3.git
2.      cd AreTomo3
3.      make exe -f makefile11 [CUDAHOME=path/cuda-xx.x]

If the compute capability of GPUs is 5.x, use makefile instead. If CUDAHOME is not provided, the default installation path of CUDA given in makefile or makefile11 will be used.

## Code of Conduct

This project adheres to the Contributor Covenant [code of conduct](https://github.com/chanzuckerberg/.github/blob/master/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [opensource@chanzuckerberg.com](mailto:opensource@chanzuckerberg.com).

## Reporting Security Issues

If you believe you have found a security issue, please responsibly disclose by contacting us at [security@chanzuckerberg.com](mailto:security@chanzuckerberg.com).
