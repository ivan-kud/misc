from adcparam import gen_signal, AdcParam

import matplotlib.pyplot as plt


N = 8
M = 2**(N+3)
G = 1
lsb = 1
harmonics = 10
method = 'Simple'
window = 'Blackman-Harris'
bins = [4, 7, 5]

# Generate signal
size = 3*M + 1
fs = 5000 * 10 ** 6
f1 = 30 * fs / M
x = gen_signal(size=size, bits=N, lsb=lsb, fs=fs, f1=f1, harmonics=harmonics,
               h1=0.95, h2=0.02, attenuation=0.90, noise=0.02, offset=0.02, bipolarity=False)

# Compute and print ADC parameters
adc = AdcParam(x=x, n=N, g=G, lsb=lsb)
adc.params_td(m=None, periods=10000)
print('Time domain:')
print('SINAD =', adc.sinad_td)
print('ENOB =', adc.enob_td)
print()
adc.params_fd(m=M, harmonics=harmonics, method=method, window=window, bins=bins)
print('Frequency domain:')
print('THD =', adc.thd_fd)
print('SNR =', adc.snr_fd)
print('SINAD =', adc.sinad_fd)
print('ENOB =', adc.enob_fd)
print()

# Check values
# Does 'energy_delta_fd' equal zero? It may not equal zero for 'Std 1241-2010' method.
if method == 'Std 1241-2010':
    assert (adc.energy_delta_fd > -adc.energies_fd[0] / 1000)\
           and (adc.energy_delta_fd < adc.energies_fd[0] / 1000),\
           'Variable "energy_delta_fd" not equals zero.'
else:
    assert (adc.energy_delta_fd > -adc.energies_fd[0] / 10000)\
           and (adc.energy_delta_fd < adc.energies_fd[0] / 10000),\
           'Variable "energy_delta_fd" not equals zero.'
# Does 'unity_fd' equal 1?
assert (adc.unity_fd > 0.9999) and (adc.unity_fd < 1.0001), 'Variable "unity_fd" not equals zero.'

# Plot
fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(x)
plt.subplot(2, 1, 2)
plt.plot(adc.Xamp_log)
plt.show()
