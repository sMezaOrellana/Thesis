import tensorflow_io as tfio
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
import scipy as sp
import scipy.signal as signal
file_name = 'pc.wav'
ENF_FILES =  os.path.join('data', file_name)

def butter_bandpass(lowcut, highcut, nyq, order=None):
    """
    Function Name: butter_bandpass
    Description:
        Function to setup butterworth bandpass filter and
        return the proper coefficients.
    Input(s):
        lowcut - low cutoff frequency
        highcut - high cutoff frequency
        nyq - nyquist rate (sample_rate / 2)
        order - filter order (optional. default = 2)
    Return(s):
        b , a - filter coefficients
    """
    # Check If Optional Arg Is None
    if order is None:
        order = 2

    # Set Bandpass Frequencies
    low = lowcut / nyq
    high = highcut / nyq

    # Determine Coefficients For Filter Setup
    b, a = sp.signal.butter(order, [low, high], btype='band')

    return b, a

def butter_bandpass_filter(data, lowcut, highcut, nyq, order=None):
    """
    Function Name: butter_bandpass_filter
    Description:
        Function to setup and filter data using a butterworth
        bandpass filter.
    Input(s):
        data - data to filter
        lowcut - low cutoff frequency
        highcut - high cutoff frequency
        nyq - nyquist rate (sample_rate / 2)
        order - order of filter (optional. default = 2)
    Return(s):
        y - filtered data
    """
    # Check If Optional Arg Is None
    if order is None: 
        order = 2

    # Get Coefficients And Filter Signal
    b, a = butter_bandpass(lowcut, highcut, nyq, order=order)
    y = sp.signal.lfilter(b, a, data)

    # Return Filtered Data
    return y

def load_wav(filename, down_sr):
	file_contents = tf.io.read_file(filename)

	wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
	wav = tf.squeeze(wav, axis=-1)
	sample_rate = tf.cast(sample_rate, dtype=tf.int64)

	wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=down_sr)
	return wav.numpy()

def main():
	down_sample = 120
	wav = load_wav(ENF_FILES,down_sample)
	wav = butter_bandpass_filter(wav,49,51, int(down_sample/2))

	# generate power spectrum
	plt.psd(wav,NFFT=int(4096),noverlap= 120, Fs=down_sample)

	plt.savefig(file_name + "PSD.png")
	plt.clf()
	# generate spectrogram

	d2, freq, t , _ = plt.specgram(wav,NFFT=int(4096/4),noverlap= 120, Fs=down_sample, cmap="jet_r")
	print(d2)
	print(len(freq))
	print(t)
	# Set the title of the plot, xlabel and ylabel
	# and display using show() function
	plt.title('Spectogram of ' + file_name)

	save_name = file_name + '.png'
	plt.savefig(save_name)
	print(ENF_FILES)

if __name__ == "__main__":
    main()