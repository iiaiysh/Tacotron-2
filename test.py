import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

wav_path = '/raid1/stephen/data/Blizzard_2012/ATrampAbroad/wav/chp56_00098.wav'
# wav_path = '/raid1/stephen/data/LJSpeech-1.1/wavs/LJ001-0001.wav'

x, sr = librosa.load(wav_path)

print(x.shape)


#
hop_length = 512
n_fft = 2048



X = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)

print(X.shape)

S = librosa.feature.melspectrogram(x, sr=sr, n_fft=n_fft, hop_length=hop_length)

print(S.shape)


import matplotlib.pyplot as plt
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time')
plt.title(f'S:{wav_path}')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

librosa.display.specshow(librosa.amplitude_to_db(X, ref=np.max), y_axis='log', x_axis='time')
plt.title(f'X:{wav_path}')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()