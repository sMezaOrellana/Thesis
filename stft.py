import numpy as np
from matplotlib import pyplot as plt
import os
import scipy as sp
import scipy.signal as signal
import librosa
file_name = 'pc.wav'
ENF_FILES =  os.path.join('data', file_name)

sig, sr = librosa.load(ENF_FILES, mono=True)
sr_n = 120
sig = librosa.resample(sig, orig_sr=sr, target_sr=sr_n)

def main():
	f,t,Z = signal.stft(sig, sr_n, nfft=4096,noverlap = 120)

	f_filt = f[ f > 49.000]
	f_filt = f_filt[f_filt < 51.000]
	print(f_filt)
	print(Z.shape)

	plt.pcolormesh(t, f, np.abs(Z), shading='gouraud')
	plt.title('STFT Magnitude')
	plt.ylabel('Frequency [Hz]')
	plt.xlabel('Time [sec]')
	save_name = 'stft' + file_name + '.png'
	plt.savefig(save_name)
	return 1

if __name__ == "__main__":
    main()