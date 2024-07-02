import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit import Model
from scipy.optimize import curve_fit, nnls
from scipy.fft import ifft
from tqdm import tqdm
from scipy.signal import savgol_filter

def txt_to_dataframe(filename):
    # Find the index of the row that starts with "Time[ns]  No_of_photons"
    start_index = 0
    end_index = 0
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if "*BLOCK 1  ( Time[ns]  No_of_photons )" in line:
                start_index = i + 1
            # if "*END" in line:
            #     end_index = i
    data = pd.read_csv(filename, sep='\t', skiprows=start_index, skipfooter = end_index)
    # Rename first column
    data.rename(columns={data.columns[0]: 'Time Photons'}, inplace=True)
    # Separate columns into two columns
    data[['Time', 'Photons']] = data['Time Photons'].str.split(expand=True)
    # Remove first column
    data = data.drop(columns=['Time Photons'])
    # Convert 'Photons' to numeric
    data['Photons'] = pd.to_numeric(data['Photons'], errors='coerce')
    data['Time'] = pd.to_numeric(data['Time'], errors='coerce')
    # Remove Nans
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data.sort_values(by='Time')
    return data

def create_dataframe_list(directory, file_startswith):
    dataframe_list = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.startswith(file_startswith) and filename.endswith(".asc"):
            df = txt_to_dataframe(os.path.join(directory, filename))
            dataframe_list.append(df)
            filenames.append(filename)
    return dataframe_list, filenames

def smooth_df(df, window_length=5, polyorder=3, title=None):
    df_smooth = df.copy()
    df_smooth['Photons'] = savgol_filter(df['Photons'], window_length=window_length, polyorder=polyorder)
    #plot smoothed data and original data
    plt.plot(df['Time'], df['Photons'], label='Original Data')
    plt.plot(df['Time'], df_smooth['Photons'], label='Smoothed Data')
    #set logy
    plt.yscale('log')
    #set ylimit
    plt.ylim(0.1)
    if title:
        plt.title(title)
    plt.legend()
    plt.show()
    return df_smooth

def prepare_data(signal, irf, window_length=None, polyorder=None):
    # Remove leading and trailing rows where 'Photons' is 0
    mask = signal['Photons'].ne(0)
    signal = signal.loc[mask.idxmax():mask[::-1].idxmax()]
    mask_irf = irf['Photons'].ne(0)
    irf = irf.loc[mask_irf.idxmax():mask_irf[::-1].idxmax()]
    
    # Make lengths equal
    min_length = min(len(signal), len(irf))
    signal = signal.iloc[:min_length]
    irf = irf.iloc[:min_length]

    # Smooth data
    if window_length and polyorder:
        signal = smooth_df(signal, window_length=window_length, polyorder=polyorder, title='Signal')
        irf = smooth_df(irf, window_length=window_length, polyorder=polyorder, title='IRF')
        
    # Normalize IRF
    irf['Photons'] = irf['Photons'] / irf['Photons'].sum()
    return signal, irf

def generate_gauss_irf(time, mu, sigma=0.01, A=10000):
    df_irf = pd.DataFrame()
    df_irf['Time'] = time
    df_irf['Photons'] = A * np.exp(-(time - mu)**2 / (2 * sigma**2))
    return df_irf

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

def single_exp_model(x, tau1, ampl1, irf, irf_shift, bg):
    irf = shift_irf(irf / np.sum(irf), irf_shift)
    ymodel = ampl1 * np.exp(-x / tau1)
    z = convolve(ymodel, irf)[:len(x)]
    return z + bg

def double_exp_model(x, tau1, ampl1, tau2, ampl2, irf, irf_shift, bg):
    irf = shift_irf(irf / np.sum(irf), irf_shift)
    ymodel1 = ampl1 * np.exp(-x / tau1)
    ymodel2 = ampl2 * np.exp(-x / tau2)
    z1 = convolve(ymodel1, irf)[:len(x)]
    z2 = convolve(ymodel2, irf)[:len(x)]
    return z1 + z2 + bg

def triple_exp_model(x, tau1, ampl1, tau2, ampl2, tau3, ampl3, irf, irf_shift, bg):
    irf = shift_irf(irf / np.sum(irf), irf_shift)
    ymodel1 = ampl1 * np.exp(-x / tau1)
    ymodel2 = ampl2 * np.exp(-x / tau2)
    ymodel3 = ampl3 * np.exp(-x / tau3)
    z1 = convolve(ymodel1, irf)[:len(x)]
    z2 = convolve(ymodel2, irf)[:len(x)]
    z3 = convolve(ymodel3, irf)[:len(x)]
    return z1 + z2 + z3 + bg

def get_weights(decay):
    weights = 1 / np.sqrt(decay + 1)
    return weights

def prepare_data(df_signal, df_irf, window_length=None, polyorder=None):
    signal = df_signal.copy()
    irf = df_irf.copy()
    
    # Remove leading and trailing zeros
    mask = signal['Photons'].ne(0)
    signal = signal.loc[mask.idxmax():mask[::-1].idxmax()]

    mask = irf['Photons'].ne(0)
    irf = irf.loc[mask.idxmax():mask[::-1].idxmax()]
    
    # Smooth data if parameters are provided
    if window_length and polyorder:
        signal['Photons'] = signal['Photons'].rolling(window=window_length, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        irf['Photons'] = irf['Photons'].rolling(window=window_length, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    
    return signal, irf

def nnls_convol_irfexp(x_data, irf, p0, signal_values):
    irf_shift, *tau0 = p0
    shifted_irf = shift_irf(irf, irf_shift)
    
    decays = [convolve(shifted_irf, exp_decay(x_data, tau)) for tau in tau0]
    decays.append(np.ones_like(x_data))  # Adding ones for the offset component
    
    # Find the minimum length among the arrays
    min_length = min(len(decays[0]), len(signal_values))
    
    # Trim all arrays to the minimum length
    decays_trimmed = [decay[:min_length] for decay in decays]
    signal_values_trimmed = signal_values[:min_length]
    
    A = np.vstack(decays_trimmed).T
    x, _ = nnls(A, signal_values_trimmed)
    y = np.dot(A, x)
    return A, x, y

def model_func(x_data, irf, p0, signal_values):
    A, x, y = nnls_convol_irfexp(x_data, irf, p0, signal_values)
    return y

def plot_reconvolution_curve_fit(times, signal, fit, irf, irf_shift, taus, amplitudes, plot_title, filename=None):
    residuals = signal - fit
    min_length_irf = min(len(times), len(irf))
    times_irf = times[:min_length_irf]
    irf_adjusted = irf[:min_length_irf]

    title_str = f"{plot_title}\nFitted IRF Shift: {np.round(irf_shift, 5)} ns\n Amplitudes {np.round(amplitudes, 5)} [photons]\n Taus {np.round(taus, 5)} [ns]\n $\chi^2$: {np.sum(residuals**2)/len(residuals)}"
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(12, 8))
    fig.tight_layout()

    ax[0].set_title(title_str)
    ax[0].plot(times, signal, 'r-', label='Signal', alpha=0.5)
    ax[0].plot(times, fit, 'b-', label='Fit (reconvolution convoluted with IRF)', alpha=0.5)
    ax[0].plot(times_irf, irf_adjusted, 'g-', label='IRF', alpha=0.5)
    ax[0].plot(times_irf, shift_irf(irf_adjusted, irf_shift), 'g--', label='Shifted IRF', alpha=0.5)
    
    reconvolution_sum = sum([amplitudes[i] * np.exp(-times / taus[i]) for i in range(len(taus))])
    ax[0].plot(times, reconvolution_sum, 'y-', label='Reconvolution', alpha=0.5, linewidth=4)
    
    for i in range(len(taus)):
        ax[0].plot(times, amplitudes[i] * np.exp(-times / taus[i]), label=f'Exponent {i+1}', alpha=0.5, linewidth=3, linestyle='--')

    ax[0].set_xlim(times.min(), times.max())
    ax[0].set_ylim(0.01)
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

def reconvolution_curve_fit(df_signal, df_irf, exp_num=1, tau_bounds=(0, np.inf), smooth=None, num_splits=10, plot_title='Exponential Decay Reconvolution', filename=None):
    if smooth:
        window_length, polyorder = smooth
        signal, irf = prepare_data(df_signal, df_irf, window_length, polyorder)
    else:
        signal, irf = prepare_data(df_signal, df_irf)
    
    x_data = signal["Time"].values
    y_data = signal["Photons"].values
    irf_values = irf["Photons"].values    
    exp_num = range(exp_num)
    
    initial_irf_shift = find_irf_shift(y_data, irf_values)
    
    tau_grid = [np.linspace(tau_bounds[0], tau_bounds[1], num_splits) for _ in exp_num]
    tau_combinations = np.array(np.meshgrid(*tau_grid)).T.reshape(-1, len(exp_num))
    
    best_fit = None
    best_popt = None
    best_chi2 = np.inf
    with tqdm(total=len(tau_combinations), desc="Fitting progress") as pbar:
        for tau_comb in tau_combinations:
            p0 = [initial_irf_shift, *tau_comb]
            try:
                weights = 1 / np.sqrt(y_data + 1)
                popt, pcov = curve_fit(lambda x, *p: model_func(x, irf_values, p, y_data), x_data, y_data, p0=p0, bounds=([-len(x_data)] + [tau_bounds[0]]*len(exp_num), [len(x_data)] + [tau_bounds[1]]*len(exp_num)), sigma=weights)
                _, _, y_fit = nnls_convol_irfexp(x_data, irf_values, popt, y_data)
                residuals = y_data - y_fit
                chi2 = np.sum((residuals / weights) ** 2)
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
    
    print(f"Optimal IRF shift: {irf_shift_opt}")
    for i, (tau, amp) in enumerate(zip(tau_opt, amplitudes)):
        print(f"Tau {i+1}: {tau}, Amplitude {i+1}: {amp}")
    print(f"Offset: {x[-1]}")
    print(f"Fitted IRF shift: {best_popt[0]}")

    plot_reconvolution_curve_fit(x_data, y_data, y_fit, irf_values, irf_shift_opt, tau_opt, amplitudes, plot_title, filename)
    
    return best_popt, pcov

def fit_model(mod, pars, signal, times, irf, weights, num_splits, bounds, model_type):
    tau1_vals = np.linspace(bounds[model_type]['tau1'][0], bounds[model_type]['tau1'][1], num_splits)
    tau2_vals = np.linspace(bounds[model_type]['tau2'][0], bounds[model_type]['tau2'][1], num_splits) if model_type in ['double', 'triple'] else [None]
    tau3_vals = np.linspace(bounds[model_type]['tau3'][0], bounds[model_type]['tau3'][1], num_splits) if model_type == 'triple' else [None]

    best_result = None
    best_chi2 = np.inf

    total_iterations = len(tau1_vals) * (len(tau2_vals) if tau2_vals[0] is not None else 1) * (len(tau3_vals) if tau3_vals[0] is not None else 1)

    with tqdm(total=total_iterations, desc='Fitting') as pbar:
        for tau1_init in tau1_vals:
            for tau2_init in (tau2_vals if tau2_vals[0] is not None else [None]):
                for tau3_init in (tau3_vals if tau3_vals[0] is not None else [None]):
                    taus = [tau1_init]
                    if model_type in ['double', 'triple']:
                        taus.append(tau2_init)
                    if model_type == 'triple':
                        taus.append(tau3_init)
                    pars['ampl1'].set(value=np.max(signal))
                    if model_type in ['double', 'triple']:
                        pars['ampl2'].set(value=np.max(signal))
                    if model_type == 'triple':
                        pars['ampl3'].set(value=np.max(signal))
                    pars['bg'].set(value=np.min(signal))
                    result = mod.fit(signal, params=pars, x=times, irf=irf, weights=weights, method='least_squares')
                    if result.redchi < best_chi2:
                        best_result = result
                        best_chi2 = result.redchi
                    pbar.update(1)
    return best_result

def plot_lmfit(times, signal, irf, result, model_type, plot_title, shift, filename=None):
    min_length_irf = min(len(times), len(irf))
    times_irf = times[:min_length_irf]
    irf_adjusted = irf[:min_length_irf]

    title_str = (f"{plot_title}\n$R^2$: {np.round(result.rsquared, 2)}, $\\chi^2_\\nu$: {np.round(result.redchi, 2)}, "
                f"$\\Delta$IRF: {np.round(shift, 2)}, $\\Delta bg$: {np.round(result.best_values['bg'], 2)}")

    if model_type == 'single':
        title_str += (f", $A_1$: {np.round(result.best_values['ampl1'], 6)}, $\\tau_1$: {np.round(result.best_values['tau1'], 6)} ns")
    elif model_type == 'triple':
        title_str += (f", $A_1$: {np.round(result.best_values['ampl1'], 6)}, $\\tau_1$: {np.round(result.best_values['tau1'], 6)} ns, "
                    f"$A_2$: {np.round(result.best_values['ampl2'], 6)}, $\\tau_2$: {np.round(result.best_values['tau2'], 6)} ns, "
                    f"$A_3$: {np.round(result.best_values['ampl3'], 6)}, $\\tau_3$: {np.round(result.best_values['tau3'], 6)} ns")
    else:  # default to double
        title_str += (f", $A_1$: {np.round(result.best_values['ampl1'], 6)}, $\\tau_1$: {np.round(result.best_values['tau1'], 6)} ns, "
                    f"$A_2$: {np.round(result.best_values['ampl2'], 6)}, $\\tau_2$: {np.round(result.best_values['tau2'], 6)} ns")

    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(12, 6))
    fig.tight_layout()

    ax[0].set_title(title_str)
    ax[0].plot(times, signal, 'r-', label='Signal', alpha=0.5)
    ax[0].plot(times, result.best_fit[:len(signal)], 'b-', label='Fit (reconvolution convoluted with IRF)', alpha=0.5)
    ax[0].plot(times_irf, irf_adjusted, 'g-', label='IRF', alpha=0.5)

    if model_type == 'single':
        ax[0].plot(times, result.best_values['ampl1'] * np.exp(-times / result.best_values['tau1']), 
                'y-', label='Reconvolution', alpha=0.5, linewidth=4)
    elif model_type == 'triple':
        ax[0].plot(times, result.best_values['ampl1'] * np.exp(-times / result.best_values['tau1']) + 
                result.best_values['ampl2'] * np.exp(-times / result.best_values['tau2']) + 
                result.best_values['ampl3'] * np.exp(-times / result.best_values['tau3']),
                'y-', label='Reconvolution', alpha=0.5, linewidth=4)
    else:  # default to double
        ax[0].plot(times, result.best_values['ampl1'] * np.exp(-times / result.best_values['tau1']) + 
                result.best_values['ampl2'] * np.exp(-times / result.best_values['tau2']), 
                'y-', label='Reconvolution', alpha=0.5, linewidth=4)

    ax[0].set_xlim(times.min(), times.max())
    ax[0].set_ylim(0.01)
    ax[0].legend()
    ax[0].set_yscale('log')
    ax[0].set_xlabel('Time (ns)')
    ax[0].set_ylabel('Photons')
    residuals = (result.best_fit[:len(signal)] - signal)
    ax[1].plot(times, result.residual, 'r-', label='Weighted', alpha=1)
    ax[1].plot(times, residuals, 'b-', label='Residuals', alpha=0.5)
    ax[1].set_ylim(-100, 100)
    ax[1].set_xlim(times.min(), times.max())
    ax[1].set_xlabel("Time (ns)")
    ax[1].set_ylabel('Residuals')
    ax[1].axhline(y=0, color='grey', linestyle='--')
    ax[0].legend()
    if filename:  
        plt.tight_layout()
        plt.savefig(filename + '.png')

def reconvolution_lmfit(df_signal, df_irf, model_type='double', bounds=None, smooth=None, num_splits=10, plot_title='Exponential Decay Reconvolution'):
    if smooth:
        window_length, polyorder = smooth
        signal, irf = prepare_data(df_signal, df_irf, window_length, polyorder)
    else:
        signal, irf = prepare_data(df_signal, df_irf)
    
    x_data = signal["Time"].values
    y_data = signal["Photons"].values
    irf_values = irf["Photons"].values
    irf_values /= np.sum(irf_values)

    initial_irf_shift = find_irf_shift(y_data, irf_values)
    irf_values = np.roll(irf_values, initial_irf_shift)
    
    if bounds is None:
        bounds = {
            'single': {'tau1': (1e-6, 0.01), 'ampl1': (1e-6, None), 'irf_shift': (-10, 10), 'bg': (-20, 20)},
            'double': {'tau1': (1e-6, 0.01), 'ampl1': (1e-6, None), 'tau2': (0.01, 1), 'ampl2': (1e-6, None), 'irf_shift': (-30, 50), 'bg': (-20, 20)},
            'triple': {'tau1': (1e-6, 0.01), 'ampl1': (1e-6, None), 'tau2': (0.01, 1), 'ampl2': (1e-6, None), 'tau3': (0.1, 2), 'ampl3': (1e-6, None), 'irf_shift': (-10, 10), 'bg': (-20, 20)}
        }

    model_funcs = {
        'single': single_exp_model,
        'double': double_exp_model,
        'triple': triple_exp_model
    }

    mod = Model(model_funcs[model_type], independent_vars=('x', 'irf'))
    pars = mod.make_params()
    for key, bound in bounds[model_type].items():
        value = (bound[0] + (bound[1] if bound[1] is not None else bound[0])) / 2
        pars[key].set(value=value, min=bound[0], max=bound[1])

    pars['irf_shift'].set(value=initial_irf_shift)

    weights = get_weights(y_data)
    best_result = fit_model(mod, pars, y_data, x_data, irf_values, weights, num_splits, bounds, model_type)

    if best_result:
        print(f"Optimal IRF shift: {best_result.best_values['irf_shift']}")
        for key in best_result.best_values:
            if key.startswith('tau') or key.startswith('ampl'):
                print(f"{key}: {best_result.best_values[key]}")
        print(f"Background: {best_result.best_values['bg']}")

        plot_lmfit(x_data, y_data, irf_values, best_result, model_type, plot_title, best_result.best_values['irf_shift'])
    
    return best_result, best_result.best_values

