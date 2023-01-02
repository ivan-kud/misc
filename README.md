## Utility for computing dynamic ADC characteristics

The module is designed to calculate the dynamic characteristics of analog-to-digital converters (ADC) in accordance with the IEEE Std 1241-2010 standard. The calculation is available by two methods: using the approximation of the signal by a sine wave and using the discrete Fourier transform (DFT) into a spectrum.

In accordance with IEEE Std 1241-2010, when using the DFT method, it is recommended not to use a window function. This prevents the signal energy from spreading across the spectrum, which increases the accuracy of calculations. However, this requires registering an integer number of periods of the input sinusoidal signal.

The calculated dynamic characteristics include:
- Effective number of bits (ENOB).
- Signal-to-noise and distortion ratio (SINAD).
- Signal-to-noise ratio (SNR) (only by the DFT method so far).
- Total harmonic distortion (THD) (only by the DFT method so far).

## Cumulative ARPU (cohort analysis)
**Project Description:** Cohort analysis of mobile applications A and B.

**Goal:** To increase revenue from app users.

**Objectives:** Analysis of cumulative ARPU for various cohorts.

**Dataset Description:** Dataset consists of two files of 1620 and 835380 rows.
