import os            # For file and directory handling
import re            # For regular expressions (extracting cycle number)
import numpy as np    # For numerical operations
import pandas as pd   # For data manipulation with DataFrames
import matplotlib.pyplot as plt  # For plotting
from matplotlib import cm         # For color maps in plots
from scipy.optimize import nnls   # For non-negative least squares optimization
# from scipy.optimize import least_squares  # For least-squares optimization
from scipy.optimize import differential_evolution  # For differential evolution optimization
from scipy.signal import savgol_filter   # For Savitzky-Golay filtering

def parse_FLIM(filename):
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

def extract_cycle_number(filename):
    """
    Extract the number at the end of the filename before the '.asc' extension.
    Assumes filenames are in the format of 'somethingNUMBER.asc'.
    
    :param filename: Filename string.
    :return: The extracted number as an integer.
    """
    match = re.search(r'c(\d+)\.asc$', filename)
    if match:
        return int(match.group(1))  # Extract the number as an integer
    else:
        return float('inf')  # Return a very large number if no match is found (this shouldn't happen)
    

def parse_FLIM_set(directory, file_startswith):
    dataframe_list = []
    filenames = []
    
    # List all files in the directory that match the given criteria
    files = [f for f in os.listdir(directory) if f.startswith(file_startswith) and f.endswith(".asc")]
    
    # Sort files by the extracted number from the filename (small to high)
    files.sort(key=lambda f: extract_cycle_number(f))
    
    # Parse each file and append the dataframe to the list
    for filename in files:
        filepath = os.path.join(directory, filename)
        df = parse_FLIM(filepath)
        dataframe_list.append(df)
        filenames.append(filename)
    
    return dataframe_list  # Returning filenames too if needed

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

def plot_FLIM_spectrum(dataframes, min_scale=1, title="Spectra Plot"):
    """
    This function plots the spectra (Photons vs. Time) for either a single dataframe or a list of dataframes.
    
    :param dataframes: A single dataframe or a list of dataframes.
    :param title: Title of the plot.
    :return: None
    """
    # Check if input is a list or a single dataframe
    if isinstance(dataframes, list):
        # Plot for a list of dataframes using a colormap
        num_dfs = len(dataframes)
        cmap = cm.get_cmap('jet', num_dfs)  # Get the 'jet' colormap
        
        plt.figure(figsize=(8, 4))
        for i, df in enumerate(dataframes):
            plt.plot(df['Time'], df['Photons'], label=f'Spectra {i+1}', color=cmap(i), alpha=0.1)
        plt.xlabel('Time [ns]')
        plt.ylabel('Photons')
        plt.yscale('log')
        plt.title(title)
        
        # Add colorbar
        norm = plt.Normalize(vmin=1, vmax=num_dfs*min_scale)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label='Minutes')
        plt.grid(True)
        plt.show()
    
    else:
        # Plot for a single dataframe (no colorbar)
        plt.figure(figsize=(8, 4))
        # apply Savitzky-Golay filter
        
        dataframes['Photons'] = savgol_filter(dataframes['Photons'], window_length=6, polyorder=3)
        
        plt.plot(dataframes['Time'], dataframes['Photons'], color='blue')
        
        plt.xlabel('Time [ns]', fontsize=14)
        plt.ylabel('Photons', fontsize=14)
        plt.yscale('log')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title(title, fontsize=16)
        plt.grid(True)
        plt.show()
def sci_notation(number, sig_fig=2):
    ret_string = "{0:.{1:d}e}".format(number, sig_fig)
    a, b = ret_string.split("e")
    b = int(b)
    return a + " * 10^" + str(b)

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
def calculate_fwhm(time, photons):
    """
    Calculate Full Width at Half Maximum (FWHM) of a peak.
    
    :param time: Array of time values.
    :param photons: Array of photon values.
    :return: FWHM value.
    """
    max_photons = np.max(photons)
    half_max = max_photons / 2.0
    indices_above_half_max = np.where(photons >= half_max)[0]
    
    if len(indices_above_half_max) >= 2:
        fwhm = time[indices_above_half_max[-1]] - time[indices_above_half_max[0]]
    else:
        fwhm = np.nan
    
    return fwhm

def calculate_integral(time, photons):
    """
    Calculate the integral of the peak using the trapezoidal rule.
    
    :param time: Array of time values.
    :param photons: Array of photon values.
    :return: The integral of the peak.
    """
    return np.trapz(photons, x=time)

def plot_histogram(dataframes, 
                    what_to_plot='max_photons', 
                    min_scale=1,
                    bad_bins=None,
                    window_length=11, 
                    polyorder=2,
                    plot_name='Plot'):
    """
    This function processes a list of dataframes, applies Savitzky-Golay filter, 
    and plots the requested quantity (max photons, FWHM, or integral of the peak).
    
    :param dataframes: List of pandas DataFrames with 'Time' and 'Photons' columns.
    :param what_to_plot: 'max_photons', 'fwhm', or 'integral'.
    :param window_length: Window length for the Savitzky-Golay filter (default: 11).
    :param polyorder: Polynomial order for the Savitzky-Golay filter (default: 2).
    :param bad_bins: List of bad bins to be interpolated.
    :param plot_name: Title of the plot.
    :return: None
    """
    result_values = []
    
    # Step 1: Process each dataframe
    for i, df in enumerate(dataframes, 1):
        # Remove baseline (assumed to be the minimum value in the 'Photons' column)
        df['Photons_baseline_corrected'] = df['Photons'] - df['Photons'].min()
        
        # Apply Savitzky-Golay filter to the baseline-corrected data
        df['Photons_smoothed'] = savgol_filter(df['Photons_baseline_corrected'], window_length, polyorder)
        
        # Calculate the requested quantity
        if what_to_plot == 'max_photons':
            color = 'blue'
            result = df['Photons_smoothed'].max()
        elif what_to_plot == 'fwhm':
            color = 'green'
            result = calculate_fwhm(df['Time'].values, df['Photons_smoothed'].values)
        elif what_to_plot == 'integral':
            color = 'red'
            result = calculate_integral(df['Time'].values, df['Photons_smoothed'].values)
        else:
            raise ValueError(f"Unknown plot option: {what_to_plot}")
        
        result_values.append(result)
    
    # Step 2: Interpolate bad bins if provided
    if bad_bins is not None:
        for bad_bin in bad_bins:
            if 0 < bad_bin < len(result_values) - 1:
                # Interpolate from the two neighbors
                result_values[bad_bin] = (result_values[bad_bin - 1] + result_values[bad_bin + 1]) / 2
            elif bad_bin == 0:
                # If it's the first bin, use the next one
                result_values[bad_bin] = result_values[bad_bin + 1]
            elif bad_bin == len(result_values) - 1:
                # If it's the last bin, use the previous one
                result_values[bad_bin] = result_values[bad_bin - 1]
    
    # Step 3: Prepare x-axis (binning)
    x_values = np.arange(1, len(result_values) + 1)  # Default numbering
    
    # Step 4: Plot the result values using a bar plot or scatter plot
    plt.figure(figsize=(20, 6))
    plt.bar(x_values, result_values, color=color, edgecolor='black')
    plt.xlabel('Minutes')
    
    # Set y-label based on what is being plotted
    if what_to_plot == 'max_photons':
        plt.ylabel('Max Photons')
    elif what_to_plot == 'fwhm':
        plt.ylabel('FWHM [ns]')
    elif what_to_plot == 'integral':
        plt.ylabel('Total Photons')
    
    # Update x-axis ticks based on bin size
    plt.xticks(x_values, labels=x_values * min_scale)
    plt.title(plot_name)
    plt.yscale('log')
    plt.grid(True)
    plt.show()

def residual_function(params, x_data, irf_values, y_data):
    irf_shift = params[0]
    taus = params[1:]
    p0 = [irf_shift] + list(taus)
    _, _, y_fit = nnls_convol_irfexp(x_data, irf_values, p0, y_data)
    residuals = (y_fit - y_data) / np.sqrt(y_fit+1)
    chi2 = np.sum(residuals ** 2)
    chi2_reduced = chi2 / (len(x_data))
    return chi2_reduced

def reconvolution_fit(FLIM_data, exp_num=1, tau_bounds=None, irf_shift_bounds=[-500, 500], smooth=None, maxiter = 1000, disp = False, workers = 1, plot_title=None, filename=None):
    df_signal = FLIM_data[0]
    df_irf = FLIM_data[1]

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

    param_bounds = [(irf_shift_bounds[0], irf_shift_bounds[1])] # Bounds for the IRF shift

    # Add bounds for the tau parameters
    param_bounds.extend(tau_bounds)

    # Global minimization using Differential Evolution
    result = differential_evolution(residual_function, bounds=param_bounds, 
                                    args=(x_data, irf_values, y_data), 
                                    strategy='best1bin', maxiter=maxiter, tol=1e-12, popsize = 5*exp_num*3, polish = True, workers = workers, disp = disp)

    best_popt = result.x
    irf_shift_opt = best_popt[0]
    tau_opt = best_popt[1:]

    A, x, y_fit = nnls_convol_irfexp(x_data, irf_values, best_popt, y_data)
    amplitudes = x[:-1]

    # Sort tau and amplitudes based on tau
    tau_opt, amplitudes = zip(*sorted(zip(tau_opt, amplitudes), key=lambda x: x[0]))

    if plot_title: 
        plot_reconvolution(x_data, y_data, y_fit, irf_values, irf_shift_opt, tau_opt, amplitudes, plot_title, filename)

    return tau_opt, amplitudes, irf_shift_opt, x[-1], result.fun

def plot_reconvolution(times, signal, fit, irf, irf_shift, taus, amplitudes, plot_title, filename=None):
    residuals = (fit-signal)/np.sqrt(fit+1)

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

    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6))
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
    ax[0].set_ylim(0.5)
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