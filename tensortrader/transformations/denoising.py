import pywt
import numpy as np
import matplotlib.pyplot as plt

def wavelet_denoising(signal : np.ndarray, 
                  thresh : float,
                  wavelet: str) -> np.ndarray:
    """Removing High Frequency Noise from a 
    timeseries using the lowpassfilter
    of the wavelet transform given a wavelet function
    and a threholds. Then reconstruct the original signal
    Args:
        signal (np.ndarray): timeseries
        thresh (float): thershold between 0.1 and 1.0
        wavelet (str): wavelet function name
    Returns:
        np.ndarray: Reconstructed timeseries
    """
                  
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    return reconstructed_signal


def plot_reconstructed_signal(prices_return: np.ndarray, 
                            denoised_prices_return: np.ndarray,
                            ticker: str ) -> None:
    """Plot original and reconstructed timeseries.
    Args:
        prices_return (np.ndarray): original timeseries
        denoised_prices_return (np.ndarray): reconstructued timeseries
        ticker (str): ticker code
    """

    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(prices_return, 
                color="b", 
                alpha=0.99, 
                label='original signal')
    ax.plot(denoised_prices_return, 
            'k', label='DWT smoothing', linewidth=2)
    ax.legend()
    ax.set_title(f'Removing High Frequency Noise with DWT for {ticker}', fontsize=18)
    ax.set_ylabel('Price Return', fontsize=16)
    ax.set_xlabel('Sample No', fontsize=16)
    plt.show()

def get_significant_max_lag_pacf(pacf_values : np.ndarray, 
                                confint: np.ndarray, 
                                lags_pacf : int ) -> int:
    """Get significant lag levels for given time series
    Args:
        pacf_values (np.ndarray): pacf_values of reconstructed signal
        confint (np.ndarray): confidential intervals
        lags_pacf (int): number of lags to analyze
    Returns:
        int: deepest significant lag level
    """

    # From https://github.com/statsmodels/statsmodels/blob/8962c7fceb458d1237d977d5da605b814f4230a0/statsmodels/graphics/tsaplots.py#L31
    lower_limit, upper_limit = confint[:, 0] - pacf_values, confint[:, 1] - pacf_values

    # Get latest lag Position where PACF is significant:
    #  First get all significant position values, greater than upper limit 
    #  or lower than lower limit. 
    #  Then reverse list and get first position of True Element
    pacf_upper = list(pacf_values > upper_limit)
    pacf_lower = list(pacf_values < lower_limit)

    pacf_upper.reverse()
    pacf_lower.reverse()

    # Try to find a significant pacf value
    # if there isn't one, then assign lags_pacf
    # as the most relevant lag position, 
    # this will make max_lag_pacf = 0, 
    # meaning there isn't any relevant correlation
    # among the given lags
    try:
        pacf_upper_position = pacf_upper.index(True)
    except:
        pacf_upper_position = lags_pacf

    try:
        pacf_lower_position = pacf_lower.index(True)
    except:
        pacf_lower_position = lags_pacf


    return lags_pacf - min(pacf_upper_position, pacf_lower_position)