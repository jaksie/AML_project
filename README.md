The project is complementary to my thesis on weather forecast verification system and is related to the latest trends in implementing ML-based post-processing for numerical weather prediction. <br>

A neural network would take a 'low-resolution' WRF output of a field map of temperature at 2 m above ground as an input and reconstruct a 'super-resolution' counterpart. Optionally, local topography could be used as an additional conditioning channel. <br>

The baseline setup would be a comparison between a simple interpolation and a U-Net (and Conditional GAN, if feasible). <br>

Training data would be constructed by artificially degrading/downsampling original WRF output. <br>

The goal is to evaluate whether deep learning downscaling is a viable alternative to the standard interpolation/regression-based approach.
