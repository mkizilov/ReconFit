import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, nnls
from scipy.fft import ifft
from tqdm import tqdm
from scipy.signal import savgol_filter
import os
from scipy.optimize import differential_evolution

def parse_FILM(filename):
    start_index = 0
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if "*BLOCK 1  ( Time[ns]  No_of_photons )" in line:
                start_index = i + 1
    data = pd.read_csv(filename, sep='\t', skiprows=start_index)
    data.rename(columns={data.columns[0]: 'Time Photons'}, inplace=True)
    data[['Time', 'Photons']] = data['Time Photons'].str.split(expand=True)
    data = data.drop(columns=['Time Photons'])
    data['Photons'] = pd.to_numeric(data['Photons'], errors='coerce')
    data['Time'] = pd.to_numeric(data['Time'], errors='coerce')
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data.sort_values(by='Time')
    return data

def parse_FILM_directory(directory, file_startswith):
    dataframe_list = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.startswith(file_startswith) and filename.endswith(".asc"):
            df = parse_FILM(os.path.join(directory, filename))
            dataframe_list.append(df)
            filenames.append(filename)
    return dataframe_list

def apply_savgol(df, window_length=5, polyorder=3, title=None):
    df_smooth = df.copy()
    df_smooth['Photons'] = savgol_filter(df['Photons'], window_length=window_length, polyorder=polyorder)
    plt.plot(df['Time'], df['Photons'], label='Original Data')
    plt.plot(df['Time'], df_smooth['Photons'], label='Smoothed Data')
    plt.yscale('log')
    plt.ylim(0.1)
    if title:
        plt.title(title)
    plt.legend()
    plt.show()
    return df_smooth

def prepare_data(df_signal, df_irf, window_length=None, polyorder=None):
    signal = df_signal.copy()
    irf = df_irf.copy()
    mask = signal['Photons'].ne(0)
    signal = signal.loc[mask.idxmax():mask[::-1].idxmax()]
    mask = irf['Photons'].ne(0)
    irf = irf.loc[mask.idxmax():mask[::-1].idxmax()]
    if window_length and polyorder:
        signal['Photons'] = signal['Photons'].rolling(window=window_length, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        irf['Photons'] = irf['Photons'].rolling(window=window_length, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    # Shift to zero
    signal['Time'] -= signal['Time'].min()
    irf['Time'] -= irf['Time'].min()
    # Make them same length
    min_length = min(len(signal), len(irf))
    signal = signal[:min_length]
    irf = irf[:min_length]
    # Normalize IRF
    irf['Photons'] /= irf['Photons'].sum()  # Normalize IRF
    return signal, irf

def generate_gauss_irf(time, mu, sigma=0.01, A=10000):
    df_irf = pd.DataFrame()
    df_irf['Time'] = time
    df_irf['Photons'] = A * np.exp(-(time - mu)**2 / (2 * sigma**2))
    df_irf['Photons'] /= df_irf['Photons'].sum()  # Normalize IRF
    return df_irf

def sci_notation(number, sig_fig=2):
    ret_string = "{0:.{1:d}e}".format(number, sig_fig)
    a, b = ret_string.split("e")
    b = int(b)
    return a + " * 10^" + str(b)

def reconvolution_fit(FILM_data, exp_num=1, tau_bounds=None, smooth=None, num_splits=None, plot_title=None, filename=None):
    df_signal = FILM_data[0].copy()
    df_irf = FILM_data[1].copy()
    if smooth:
        window_length, polyorder = smooth
        signal, irf = prepare_data(df_signal, df_irf, window_length, polyorder)
    else:
        signal, irf = prepare_data(df_signal, df_irf)
    
    x_data = signal["Time"].values
    y_data = signal["Photons"].values
    irf_values = irf["Photons"].values

    if tau_bounds is None:
        tau_bounds = [(1e-12, 1)] * exp_num
    
    # Modify tau_bounds to replace any 0 with 1e-12
    tau_bounds = [(max(bound[0], 1e-12), max(bound[1], 1e-12)) for bound in tau_bounds]

    if num_splits is None:
        num_splits = [10] * exp_num
    
    initial_irf_shift = find_irf_shift(y_data, irf_values)

    # Generate logarithmic grid for tau values
    tau_grids = [np.logspace(np.log10(tau_bounds[i][0]), np.log10(tau_bounds[i][1]), num_splits[i]) for i in range(exp_num)]
    tau_combinations = np.array(np.meshgrid(*tau_grids)).T.reshape(-1, exp_num)
    
    # Count the number of valid tau combinations where tau1 > tau2 > tau3
    valid_combinations = [tau_comb for tau_comb in tau_combinations if np.all(np.diff(tau_comb) < 0)]
    
    best_fit = None
    best_popt = None
    best_chi2 = np.inf

    with tqdm(total=len(valid_combinations), desc="Fitting progress") as pbar:
        for tau_comb in valid_combinations:
            p0 = [initial_irf_shift, *tau_comb]
            try:
                weights = get_weights(y_data)
                popt, pcov = curve_fit(lambda x, *p: model_func(x, irf_values, p, y_data), x_data, y_data, p0=p0, bounds=([-len(x_data)] + [tau_bounds[i][0] for i in range(exp_num)], [len(x_data)] + [tau_bounds[i][1] for i in range(exp_num)]), sigma=weights)
                _, _, y_fit = nnls_convol_irfexp(x_data, irf_values, popt, y_data)
                residuals = y_data - y_fit
                chi2 = np.sum((residuals / weights) ** 2) / (y_data.size - len(popt))
                if chi2 < best_chi2:
                    best_chi2 = chi2
                    # best_fit = y_fit
                    best_popt = popt
            except RuntimeError:
                continue
            pbar.update(1)
    
    irf_shift_opt = best_popt[0]
    tau_opt = best_popt[1:]
    
    A, x, y_fit = nnls_convol_irfexp(x_data, irf_values, best_popt, y_data)
    amplitudes = x[:-1]
    
    # Sort tau and amplitudes based on tau
    tau_opt, amplitudes = zip(*sorted(zip(tau_opt, amplitudes), key=lambda x: x[0]))
    
    print(f"IRF shift: {irf_shift_opt}")
    print(f"Offset: {x[-1]}")
    offset = x[-1]
    for i, (tau, amp) in enumerate(zip(tau_opt, amplitudes)):
        print(f"Tau {i+1}: {tau}, Amplitude {i+1}: {amp}")
    if plot_title: 
        plot_reconvolution(x_data, y_data, y_fit, irf_values, irf_shift_opt, tau_opt, amplitudes, plot_title, filename)
    
    return tau_opt, amplitudes, irf_shift_opt, offset, best_chi2





def convolve(x, h):
    X = np.fft.fft(x, n=len(x) + len(h) - 1)
    H = np.fft.fft(h, n=len(x) + len(h) - 1)
    xch = np.real(np.fft.ifft(X * H))
    return xch[:len(x)]

def find_irf_shift(signal, irf):
    shift = np.argmax(signal) - np.argmax(irf)
    return shift

def shift_irf(irf, irf_shift):
    n = len(irf)
    channel = np.arange(n)
    irf_shifted = (1 - irf_shift + np.floor(irf_shift)) * irf[np.fmod(np.fmod(channel - np.floor(irf_shift), n) + n, n).astype(int)] + (irf_shift - np.floor(irf_shift)) * irf[np.fmod(np.fmod(channel - np.ceil(irf_shift), n) + n, n).astype(int)]
    return irf_shifted

def exp_decay(time, tau):
    return np.exp(-time / tau)

def nnls_convol_irfexp(x_data, irf, p0, signal_values):
    irf_shift, *tau0 = p0
    shifted_irf = shift_irf(irf, irf_shift)
    decays = [convolve(shifted_irf, exp_decay(x_data, tau)) for tau in tau0]
    decays.append(np.ones_like(x_data))  # Adding a constant offset term
    min_length = min(len(decays[0]), len(signal_values))
    decays_trimmed = [decay[:min_length] for decay in decays]
    signal_values_trimmed = signal_values[:min_length]
    A = np.vstack(decays_trimmed).T
    x, _ = nnls(A, signal_values_trimmed)
    y = np.dot(A, x)
    return A, x, y

def model_func(x_data, irf, p0, signal_values):
    _, _, y = nnls_convol_irfexp(x_data, irf, p0, signal_values)
    return y

def residual_function(params, x_data, irf_values, y_data):
    irf_shift = params[0]
    taus = params[1:]
    p0 = [irf_shift] + list(taus)
    
    _, _, y_fit = nnls_convol_irfexp(x_data, irf_values, p0, y_data)
    # residuals = y_data - y_fit
    # weights = get_weights(y_data)
    # chi2 = np.sum((residuals / weights) ** 2)
    
    residuals = (y_fit - y_data) / np.sqrt(y_fit)
    # residuals = residuals / get_weights(y_data)
    chi2 = np.sum(residuals ** 2)
    chi2_reduced = chi2 / (len(x_data))
    return chi2_reduced

def reconvolution_fit(FILM_data, exp_num=1, tau_bounds=None, smooth=None, plot_title=None, filename=None):
    df_signal = FILM_data[0]
    df_irf = FILM_data[1]
    
    # Data preparation
    if smooth:
        window_length, polyorder = smooth
        signal, irf = prepare_data(df_signal, df_irf, window_length, polyorder)
    else:
        signal, irf = prepare_data(df_signal, df_irf)
    
    x_data = signal["Time"].values
    y_data = signal["Photons"].values
    irf_values = irf["Photons"].values

    # Define tau bounds if not provided
    if tau_bounds is None:
        tau_bounds = [(1e-12, 2)] * exp_num

    tau_bounds = [(max(bound[0], 1e-12), max(bound[1], 1e-12)) for bound in tau_bounds]

    # Define initial IRF shift and bounds for the parameters
    initial_irf_shift = find_irf_shift(y_data, irf_values)
    param_bounds = [(-30, 30)]  # Bounds for the IRF shift
    
    # Add bounds for the tau parameters
    param_bounds.extend(tau_bounds)
    
    # Global minimization using Differential Evolution
    result = differential_evolution(residual_function, bounds=param_bounds, 
                                    args=(x_data, irf_values, y_data), 
                                    strategy='best1bin', maxiter=1000, tol=1e-12, polish = True)
    
    best_popt = result.x
    irf_shift_opt = best_popt[0]
    tau_opt = best_popt[1:]

    A, x, y_fit = nnls_convol_irfexp(x_data, irf_values, best_popt, y_data)
    amplitudes = x[:-1]

    # Sort tau and amplitudes based on tau
    tau_opt, amplitudes = zip(*sorted(zip(tau_opt, amplitudes), key=lambda x: x[0]))

    # Print the optimized parameters
    print(f"Optimized IRF shift: {irf_shift_opt}")
    print(f"Offset: {x[-1]}")
    for i, (tau, amp) in enumerate(zip(tau_opt, amplitudes)):
        print(f"Tau {i+1}: {tau}, Amplitude {i+1}: {amp}")
    
    if plot_title: 
        plot_reconvolution(x_data, y_data, y_fit, irf_values, irf_shift_opt, tau_opt, amplitudes, plot_title, filename)

    return tau_opt, amplitudes, irf_shift_opt, x[-1], result.fun


def plot_reconvolution(times, signal, fit, irf, irf_shift, taus, amplitudes, plot_title, filename=None):
    residuals = (fit-signal)/np.sqrt(fit)
    # weighted_residuals = residuals / get_weights(signal)
    
    weighted_residuals = residuals
    min_length_irf = min(len(times), len(irf))
    times_irf = times[:min_length_irf]
    irf_adjusted = irf[:min_length_irf]
    taus_scinot = [sci_notation(tau) for tau in taus]
    amplitudes_scinot = [sci_notation(amp) for amp in amplitudes] 
    # Scale the IRF to the signal
    irf_adjusted *= np.sum(signal) / np.sum(irf_adjusted)
    
    # Calculate chi2 and reduced chi2
    chi2 = np.sum((residuals ** 2))  # Chi-squared sum
    DOF_chi = len(times) - len(taus) - 1  # Degrees of freedom for chi-squared
    chi2_reduced = chi2 / DOF_chi  # Reduced chi-squared
    
    title_str = (f"{plot_title}\nIRF Shift: {np.round(irf_shift, 3)} channels\n"
                 f"Amplitudes: {amplitudes_scinot} [photons]\n"
                 f"Taus: {taus_scinot} [ns]\n"
                 f"$\\chi^2_\\nu$: {np.round(chi2_reduced, 3)}")
    
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(12, 8))
    fig.tight_layout()

    ax[0].set_title(title_str)
    ax[0].plot(times, signal, 'r-', label='Signal', alpha=0.5)
    ax[0].plot(times, fit, 'b-', label='Fit', alpha=0.5)
    ax[0].plot(times_irf, irf_adjusted, 'g-', label='IRF', alpha=0.5)
    ax[0].plot(times_irf, shift_irf(irf_adjusted, irf_shift), 'g--', label='Shifted IRF', alpha=0.5)
    
    
    reconvolution_sum = sum([amplitudes[i] * np.exp(-times / taus[i]) for i in range(len(taus))])
    ax[0].plot(times, reconvolution_sum, 'y-', label='Sum of exponents', alpha=0.5, linewidth=4)
    
    for i in range(len(taus)):
        ax[0].plot(times, amplitudes[i] * np.exp(-times / taus[i]), label=f'Exponent {i+1}', alpha=0.5, linewidth=3, linestyle='--')

    ax[0].set_xlim(times.min(), times.max())
    ax[0].set_ylim(0.1)
    ax[0].legend()
    ax[0].set_yscale('log')
    ax[0].set_xlabel('Time (ns)')
    ax[0].set_ylabel('Photons')
    
    ax[1].plot(times, residuals, 'b-', label='Residuals', alpha=0.5)
    ax[1].set_xlim(times.min(), times.max())
    ax[1].set_xlabel("Time (ns)")
    ax[1].set_ylabel('Residuals')
    ax[1].axhline(y=0, color='grey', linestyle='--')
    ax[1].legend()
    
    plt.tight_layout()
    if filename:
        plt.savefig('plots/' + filename + '.png')
    plt.show()
