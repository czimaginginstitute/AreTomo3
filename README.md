# AreTomo3
AreTomo3 is a multi-GPU accelerated software package that enables real-time fully automated reconstruction of cryoET tomograms in parallel with cryoET data collection. Integrating MotionCor3, AreTomo2, and GCtfFind in a single application, AreTomo3 has established an autonomous preprocessing pipeline that, whenever a new tilt series is collected, can activate and repeat itself from correction of beam induced motion and assembling tilt series to CTF estimation and correction, tomographic alignment, and 3D reconstruction throughout entire session of data collection without human intervention. The end results include not only tomograms but also a rich set of alignment parameters to bootstrap subtomogram averaging. Our test shows that AreTomo3 can catch up to 9-target PACE Tomo data collection with 4 NVidia RTX A6000 GPUs when it was configured to perform 2D local motion correction on movies and 3D local motion correction on tilt series. The offline testing shows that AreTomo3 runs faster than the data collection. As a result, GPU resources can be shared with other tasks to expand the preprocessing capacity. Tomogram denoising and particle picking now can run concurrently with AreTomo3 to maximize the preprocessing workflow.

## Installation
AreTomo3 is developed on Linux platform equipped with at least one Nvidia GPU card. To compile from the source, follow the steps below:

1.      git clone https://github.com/czimaginginstitute/AreTomo3.git
2.      cd AreTomo3
3.      make exe -f makefile11 [CUDAHOME=path/cuda-xx.x]

If the compute capability of GPUs is 5.x, use makefile instead. If CUDAHOME is not provided, the default installation path of CUDA given in makefile or makefile11 will be used.

## Code of Conduct

This project adheres to the Contributor Covenant [code of conduct](https://github.com/chanzuckerberg/.github/blob/main/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [opensource@chanzuckerberg.com](mailto:opensource@chanzuckerberg.com).

## Reporting Security Issues

If you believe you have found a security issue, please responsibly disclose by contacting us at [security@chanzuckerberg.com](mailto:security@chanzuckerberg.com).
