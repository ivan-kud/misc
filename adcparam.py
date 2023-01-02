import math

import numpy as np
from scipy import signal, optimize


def gen_signal(size, bits, lsb, fs, f1, harmonics=10, h1=0.95, h2=0.02,
               attenuation=0.90, noise=0.02, offset=0.02, bipolarity=True):
    """Generates digitized signal data
    size - number of digitized samples (signal length)
    bits - number of ADC bits
    lsb - ADC least significant bit. Units of lsb are usually LSB (ADC codes), V or mV.
    fs - ADC sampling frequency
    f1 - fundamental frequency
    harmonics - number of the highest order harmonic generated (integer; minimum 0)
    h1 - first harmonic excursion to fsr ratio
    h2 - second harmonic excursion to fsr ratio
    attenuation - next harmonic to previous harmonic ratio
    noise - standard deviation of Gaussian noise to fsr ratio
    offset - offset to fsr ratio (from -0.5 to 0.5)
    bipolarity - signal bipolarity flag
    """
    rng = np.random.default_rng(1234)  # rng initialization
    fsr = lsb * 2**bits  # ADC full-scale range
    time = np.arange(size) / fs  # time vector

    # Amplitude of harmonics
    a1 = h1 * fsr / 2
    a2 = h2 * fsr / 2
    if harmonics >= 2:
        an = np.append(np.array(a1, 'float64'),
                       a2 * (attenuation ** np.arange(harmonics-1)))
    elif harmonics >= 1:
        an = np.array(a1, 'float64')
    else:
        an = np.array([], 'float64')

    # Offset and RMS noise
    offset = offset * fsr if bipolarity else offset*fsr + fsr/2
    noise = noise * fsr

    # Digitized signal
    x = ((
        np.outer(np.sin(2 * np.pi * f1 * time), an).sum(1)  # harmonics
        + rng.normal(offset, noise, time.shape[0])  # offset and noise
         ) / lsb).round() * lsb  # round to lsb (digitization)
    
    # Crop signal
    if bipolarity:
        x[x < -fsr/2] = -fsr/2  # bottom crop
        x[x > fsr/2 - lsb] = fsr/2 - lsb  # top crop
    else:
        x[x < 0] = 0  # bottom crop
        x[x > fsr - lsb] = fsr - lsb  # top crop

    return x


class AdcParam:
    """ADC parameters class
    """
    def __init__(self, x, n, g=1, lsb=1):
        # Signal parameters
        self.x = x  # Signal record
        self.bipolarity = True if self.x.min() < 0 else False  # The signal x is bipolar if negative values are found
        self.Xamp = None
        self.Xamp_log = None
        self.sig_range_td = None
        self.sig_range_fd = None
        self.indices = None
        self.energies_td = None
        self.rms_values_td = None
        self.energies_fd = None
        self.rms_values_fd = None

        # Static ADC parameters
        self.n = n  # Number of ADC bits
        self.g = g  # ADC static gain
        self.lsb = lsb  # ADC least significant bit. Unit of lsb is usually LSB (ADC codes), V or mV.
        self.fsr = self.lsb * 2**self.n  # ADC full scale range
        self.min_adc_value = - self.lsb * 2**(self.n - 1) if self.bipolarity else 0
        self.max_adc_value = self.min_adc_value + self.fsr - self.lsb

        # The ADC is in overvoltage if extreme values of signal x are found (for example: 0 and 255 for 8-bit ADC).
        self.overvoltage = np.sum((self.x == self.min_adc_value) | (self.x == self.max_adc_value)) != 0

        # Dynamic ADC parameters in time domain
        self.thd_td = None
        self.snr_td = None
        self.sinad_td = None
        self.enob_td = None

        # Dynamic ADC parameters in frequency domain
        self.thd_fd = None
        self.snr_fd = None
        self.sinad_fd = None
        self.enob_fd = None

        # Test parameters
        self.energy_delta_td = None
        self.unity_td = None
        self.energy_delta_fd = None
        self.unity_fd = None

    def params_td(self, m=None, periods=10000):
        """Computes ADC parameters based on time domain calculations.
        Method description:

        Input parameters:

        Output parameters:

        """
        # Number of sequential samples in the record to do DFT
        m = self.x.shape[0] if m is None else m

        if self.x.shape[0] < m:
            raise ValueError(f'Signal record length is not enough. The minimum length equals m ({m}).')

        # Subtract FSR/2 if signal is unipolar. This decreases spectrum DC component.
        if not self.bipolarity:
            self.x -= self.fsr/2

        # Estimation of fundamental frequency
        Xamp_td = np.abs(np.fft.rfft(self.x))
        idx = np.argmax(Xamp_td[1:]) + 1
        freq_est = idx / m

        # Due to standard at least 5 periods are necessary for fitting.
        # Estimation of number of sine periods in the signal.
        max_periods_est = idx if m is None else math.floor(idx * self.x.shape[0] / m)
        periods = min(periods, max_periods_est)

        # Due to standard we can increase accuracy of fitting by using integer number of sine cycles.
        # Estimation of number of sequential samples in the record.
        max_samples_est = m if m is None else min(round(max_periods_est / freq_est), self.x.shape[0])
        samples = min(round(periods / freq_est), self.x.shape[0])

        # Estimation of amplitude and offset
        amp_est = np.std(self.x[:max_samples_est]) * math.sqrt(2)
        offset_est = np.mean(self.x[:max_samples_est])

        # Function to fit
        def fit_func(t, a, w, p, c):
            return a*np.sin(w*t + p) + c

        # Fitting
        popt, pcov = optimize.curve_fit(
            f=fit_func,
            xdata=np.arange(samples),
            ydata=self.x[:samples],
            p0=np.array([amp_est, 2. * np.pi * freq_est, 0., offset_est]),
            )

        # Residuals of (x - fit_func)
        residuals = self.x - fit_func(np.arange(self.x.shape[0]), *popt)

        # Fundamental frequency
        freq = np.abs(popt[1] / (2.*np.pi))

        # New value of input waveform frequency to set on generator at the next iteration
        # freq_next = Fi_generator + (Fi_coherence - freq)

        # Number of sine periods and number of samples in the record
        if not np.isnan(freq):
            max_periods = math.floor(freq * self.x.shape[0])
            max_samples = min(round(max_periods / freq), self.x.shape[0])
        else:
            max_periods = max_periods_est
            max_samples = max_samples_est

        # Signal offset
        offset = popt[3]  # through fitting (higher accuracy, lower stability)
        offset_est = np.mean(self.x[:max_samples_est])  # without fitting (lower accuracy, higher stability)

        # RMS value of DC
        rms_dc = np.abs(offset)  # through fitting
        rms_dc_est = np.abs(offset_est)  # without fitting

        # RMS value of the first harmonic
        rms_sig = popt[0] / math.sqrt(2)  # through fitting
        rms_sig_est = np.std(self.x[:max_samples_est] - offset_est)  # without fitting

        # RMS value of NAD
        rms_nad = np.std(residuals)

        # Total RMS
        rms_total = rms_dc + rms_sig + rms_nad

        # All RMS values
        self.rms_values_td = [rms_total, rms_dc, rms_sig, rms_nad, None, None, None]

        # First harmonic amplitude excursion to FSR rate
        sig_range = 2 * math.sqrt(2) * rms_sig * self.g / self.fsr  # through fitting
        sig_range_est = 2 * math.sqrt(2) * rms_sig_est * self.g / self.fsr  # without fitting
        self.sig_range_td = sig_range

        # ENOB (Effective Number of Bits)
        self.enob_td = math.log2(self.fsr / (self.g * rms_nad * math.sqrt(12)))

        # SINAD (Signal to Noise and Distortion Ration)
        self.sinad_td = rms_sig / rms_nad

    def params_fd(self, m=None, harmonics=10, method='Simple', window='Blackman-Harris', bins=(4, 7, 5)):
        """Computes ADC parameters based on frequency domain calculations.
        Method description:

        Input parameters:
        m - number of sequential ADC samples in a signal record that used for FFT. If there are k parts of m samples in
            signal x then method computes one average spectrum of k spectra.
            Thus only k*m samples of signal x is used and tail is dropped.
        harmonics - number of the highest order harmonic (including fundamental frequency) used to compute ADC
            parameters.
        method - method of spectrum component definition. Should be 'Simple', 'Std 1241-2010' or 'Smart'.
            'Simple' method doesn't take into account NAD under harmonic bins in spectrum.
            'Std 1241-2010' method adds average NAD under harmonic bins to overall NAD, but doesn't subtract it from the
            harmonics.
            'Smart' method does both: subtract average NAD from harmonics and adds it to overall NAD.
        window - window function used for FFT computation. Should be 'Blackman-Harris', 'Kaiser' or None. None is for
            coherent signal and means that no window function is used.
        bins - list of number of bins for spectral components: DC, first harmonic and other harmonics respectively.
            If it equals None than bins are detected automatically.

        Output parameters:

        """
        # Number of sequential samples in the record to do DFT
        m = 2**(self.n+2) if m is None else m

        if self.x.shape[0] < m:
            raise ValueError(f'Signal record length is not enough. The minimum length equals m ({m}).')

        # Subtract FSR/2 if signal is unipolar. This decreases spectrum DC component.
        if not self.bipolarity:
            self.x -= self.fsr/2

        # k is the number of data records of m points each
        k = math.floor(self.x.shape[0] / m)

        # Form KxM matrix of signal
        self.x = self.x[:k * m].reshape(k, m)

        # Window and order (number of coefficients) of it
        if window is None:  # for coherent method
            pass
        elif window == 'Blackman-Harris':
            # Periodic form of minimum four-term Blackman-Harris window
            w = signal.windows.blackmanharris(m, False)
            l = 3  # order
        elif window == 'Kaiser':
            # Periodic form of Kaiser window with beta 38
            w = signal.windows.kaiser(m, 38, False)
            l = 7  # order equivalent
        else:
            raise ValueError(f'Window name "{window}" is not available.')

        # Maximum number of bins for spectrum components. NNPG.
        if window is None:  # for coherent method
            # Maximum number of bins for DC component (integer, minimum 1, default 10)
            max_dc_bins = 10
            # Maximum number of bins for first harmonic (odd integer, minimum 1, default 19)
            max_sig_bins = 19
            # Maximum number of bins for other harmonics (odd integer, minimum 1, default 11)
            max_hrm_bins = 11
        else:  # for windowed method
            # Maximum number of bins for DC component (integer, minimum 1, default l+1=4)
            max_dc_bins = l + 1
            # Maximum number of bins for first harmonic (odd integer, minimum 1, default 2*l+1=7)
            max_sig_bins = 2*l + 1
            # Maximum number of bins for other harmonics (odd integer, minimum 1, default 2*l-1=5)
            max_hrm_bins = 2*l - 1
            
            # Normalized noise power gain
            nnpg = np.sum(w**2) / m
            
        # Due to Parseval relation, compute an average value of absolute amplitude spectrum.
        # Normalize it by m and divide by sqrt(2), to express spectrum as RMS values.
        if window is None:
            self.Xamp = np.mean(np.abs(np.fft.rfft(self.x, norm='forward')), axis=0) / math.sqrt(2) # doesn't work for 32-bit version of numpy
            # self.Xamp = np.mean(np.abs(np.fft.rfft(self.x)), axis=0) / (math.sqrt(2)*m)
        else:
            self.Xamp = np.mean(np.abs(np.fft.rfft(self.x * w, norm='forward')), axis=0) / (math.sqrt(2*nnpg)) # doesn't work for 32-bit version of numpy
            # self.Xamp = np.mean(np.abs(np.fft.rfft(self.x * w)), axis=0) / (math.sqrt(2*nnpg)*m)

        # Due to spectrum symmetry multiply it by 2
        self.Xamp = 2 * self.Xamp
        self.Xamp[0] = self.Xamp[0] / 2
        if m % 2 == 0:
            self.Xamp[-1] = self.Xamp[-1] / 2
        
        # Number of spectrum bins for real signal (with Nyquist frequency)
        m_real = self.Xamp.shape[0]

        # Threshold value to separate spectrum components
        Xamp_threshold = np.median(self.Xamp)

        # Amplitude spectrum in log scale
        self.Xamp_log = self.Xamp / self.Xamp.max()  # normalization to 1
        self.Xamp_log = 20 * np.log10(self.Xamp_log)  # conversion to dB

        # Index of DC component tail bin.
        # We should reserve at least one bin for the first harmonic.
        if bins is None:
            dc_up_idx = 0
            while((dc_up_idx + 1 <= m_real - 2)
                  and (dc_up_idx + 1 <= max_dc_bins - 1)
                  and (self.Xamp[dc_up_idx + 1] > Xamp_threshold)):
                dc_up_idx += 1
        else:
            dc_up_idx = min(bins[0] - 1, m_real - 2)

        # Index of the first harmonic
        sig_idx = int(np.argmax(self.Xamp[dc_up_idx+1:]))
        sig_idx = sig_idx + dc_up_idx + 1

        # Lower and upper indices of the first harmonic
        if bins is None:
            sig_low_idx = sig_idx
            while((sig_low_idx - 1 > dc_up_idx)
                  and (sig_low_idx - 1 > sig_idx - max_sig_bins/2)
                  and (self.Xamp[sig_low_idx - 1] > Xamp_threshold)):
                sig_low_idx -= 1
            sig_up_idx = sig_idx
            while((sig_up_idx + 1 <= m_real - 1)
                  and (sig_up_idx + 1 < sig_idx + max_sig_bins/2)
                  and (self.Xamp[sig_up_idx + 1] > Xamp_threshold)):
                sig_up_idx += 1
        else:
            sig_low_idx = max(sig_idx - int((bins[1]-1)/2), dc_up_idx + 1)
            sig_up_idx = min(sig_idx + int((bins[1]-1)/2), m_real - 1)

        # Indices of other harmonics (with aliases).
        # Necessary to check:
        # - harmonic centres must not be overlapped by each other and DC component
        #   (if overlapped, another fundamental frequency of test signal should be used;
        # - if some harmonic is at Nyquist frequency, warn would be, because
        #   computation accuracy of frequency domain method decreases;
        # - harmonic indexes can be estimated better by taking into account of first harmonic center
        #   which can be between two bins.
        h = np.append(np.arange(-harmonics, -1), np.arange(2, harmonics + 1))
        hrm_idx = h * sig_idx % m
        hrm_idx_all = np.zeros((harmonics - 1, 3), int)
        for i in range(harmonics - 1):
            if (hrm_idx[harmonics + i - 1] >= 0) and (hrm_idx[harmonics + i - 1] <= m_real - 1):
                hrm_idx_all[i, 1] = hrm_idx[harmonics + i - 1]
            else:
                hrm_idx_all[i, 1] = hrm_idx[harmonics - i - 2]

        # Lower and upper indices of other harmonics
        for i in range(harmonics - 1):
            if bins is None:
                hrm_idx_all[i, 0] = hrm_idx_all[i, 1]
                while((hrm_idx_all[i, 0] - 1 > dc_up_idx)
                      and (hrm_idx_all[i, 0] - 1 > hrm_idx_all[i, 1] - max_hrm_bins/2)
                      and (self.Xamp[hrm_idx_all[i, 0] - 1] > Xamp_threshold)):
                    hrm_idx_all[i, 0] -= 1
                hrm_idx_all[i, 2] = hrm_idx_all[i, 1]
                while((hrm_idx_all[i, 2] + 1 <= m_real - 1)
                      and (hrm_idx_all[i, 2] + 1 < hrm_idx_all[i, 1] + max_hrm_bins/2)
                      and (self.Xamp[hrm_idx_all[i, 2] + 1] > Xamp_threshold)):
                    hrm_idx_all[i, 2] += 1
            else:
                hrm_idx_all[i, 0] = max(hrm_idx_all[i, 1] - int((bins[2]-1)/2), dc_up_idx + 1)
                hrm_idx_all[i, 2] = min(hrm_idx_all[i, 1] + int((bins[2]-1)/2), m_real - 1)
            
            # Retain only unused indices
            unused_idx = set(range(hrm_idx_all[i, 0], hrm_idx_all[i, 2] + 1)) - set(range(sig_low_idx, sig_up_idx + 1))
            for j in range(i):
                unused_idx = unused_idx - set(range(hrm_idx_all[j, 0], hrm_idx_all[j, 2] + 1))
            if len(unused_idx) == 0:
                hrm_idx_all[i, 0] = 0
                hrm_idx_all[i, 2] = 0
            else:
                hrm_idx_all[i, 0] = min(unused_idx)
                hrm_idx_all[i, 2] = max(unused_idx)

        # Concatenate indices of DC, first harmonic and other harmonics
        self.indices = np.append(np.array([[0, 0, dc_up_idx], [sig_low_idx, sig_idx, sig_up_idx]]), hrm_idx_all, 0)

        # Number of bins (out of m) for the first harmonic
        num_bins_sig = 2 * (sig_up_idx-sig_low_idx+1)
        if (m % 2 == 0) and (sig_up_idx == m_real - 1):  # correct bin of Nyquist frequency
            num_bins_sig -= 1

        # Number of bins (out of m) for DC and NAD
        num_bins_dc = 2 * (dc_up_idx+1) - 1
        num_bins_nad = m - num_bins_dc - num_bins_sig

        # Total energy
        energy_total = np.sum(self.Xamp**2) + self.Xamp[0]**2

        # Energy of DC for 'Simple' method
        energy_dc = np.sum(self.Xamp[:dc_up_idx + 1]**2) + self.Xamp[0]**2

        # Energy of first harmonic for 'Simple' method
        energy_sig = np.sum(self.Xamp[sig_low_idx:sig_up_idx + 1]**2)

        # Energy of NAD (Noise and Distortion) for 'Simple' method
        energy_nad = energy_total - energy_dc - energy_sig

        # Average NAD energy per bin (out of m bins)
        energy_nad_per_bin = energy_nad / num_bins_nad

        # Energy of harmonic distortion
        energy_hrm = np.zeros(harmonics - 1, float)
        for i in range(harmonics - 1):
            if hrm_idx_all[i, 0] != 0:
                energy_hrm[i] = np.sum(self.Xamp[hrm_idx_all[i, 0]:hrm_idx_all[i, 2] + 1]**2)
        energy_hrm_total = np.sum(energy_hrm)

        # Energy correction (method specific)
        if method == 'Simple':
            pass
        elif method == 'Std 1241-2010':
            # Energy of NAD
            energy_nad = m * energy_nad_per_bin
        elif method == 'Smart':
            # Energy of NAD
            energy_nad = m * energy_nad_per_bin
            
            # Energy of DC (preventing negative and zero values)
            energy_dc = max(energy_dc - num_bins_dc*energy_nad_per_bin,
                            num_bins_dc * self.Xamp[1:m_real-1].min() / 2)
            
            # Energy of the first harmonic (preventing negative and zero values)
            energy_sig = max(energy_sig - num_bins_sig*energy_nad_per_bin,
                             num_bins_sig * self.Xamp[1:m_real-1].min() / 2)
        else:
            raise ValueError(f'Method "{method}" is not available.')

        # Energy of noise
        energy_noise = energy_nad - energy_hrm_total

        # All energies to return
        self.energies_fd = [energy_total, energy_dc, energy_sig, energy_nad, energy_hrm_total, energy_noise,
                            list(energy_hrm)]

        # RMS value of whole signal
        rms_total = math.sqrt(energy_total)

        # RMS value of harmonic distortion
        rms_hrm = np.sqrt(energy_hrm)
        rms_hrm_total = math.sqrt(energy_hrm_total)

        # RMS value of NAD
        rms_nad = math.sqrt(energy_nad)

        # RMS value of DC
        rms_dc = math.sqrt(energy_dc)

        # RMS value of the first harmonic
        rms_sig = math.sqrt(energy_sig)

        # RMS value of noise
        rms_noise = math.sqrt(energy_noise)

        # All RMS values
        self.rms_values_fd = [rms_total, rms_dc, rms_sig, rms_nad, rms_hrm_total, rms_noise, list(rms_hrm)]

        # First harmonic amplitude excursion to FSR rate
        self.sig_range_fd = 2 * math.sqrt(2) * rms_sig * self.g / self.fsr

        # THD (Total Harmonic Distortion)
        self.thd_fd = math.sqrt(energy_hrm_total / energy_sig)

        # ENOB (Effective Number of Bits)
        self.enob_fd = math.log2(self.fsr / (self.g * rms_nad * math.sqrt(12)))

        # SINAD (Signal to Noise and Distortion Ration)
        self.sinad_fd = rms_sig / rms_nad

        # SNR (Signal to Noise Ratio)
        self.snr_fd = rms_sig / rms_noise

        # Parameters for checking
        self.energy_delta_fd = energy_total - energy_dc - energy_sig - energy_nad
        self.unity_fd = self.sinad_fd**2 * (self.thd_fd**2 + 1/self.snr_fd**2)
