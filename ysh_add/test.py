import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import re
#
# wav_path = '/raid1/stephen/data/Blizzard_2012/ATrampAbroad/wav/chp56_00098.wav'
# # wav_path = '/raid1/stephen/data/LJSpeech-1.1/wavs/LJ001-0001.wav'
#
# x, sr = librosa.load(wav_path)
#
# print(x.shape)
#
#
# #
# hop_length = 512
# n_fft = 2048
#
#
#
# X = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
#
# print(X.shape)
#
# S = librosa.feature.melspectrogram(x, sr=sr, n_fft=n_fft, hop_length=hop_length)
#
# print(S.shape)
#
#
# import matplotlib.pyplot as plt
# librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time')
# plt.title(f'S:{wav_path}')
# plt.colorbar(format='%+2.0f dB')
# plt.tight_layout()
# plt.show()
#
# librosa.display.specshow(librosa.amplitude_to_db(X, ref=np.max), y_axis='log', x_axis='time')
# plt.title(f'X:{wav_path}')
# plt.colorbar(format='%+2.0f dB')
# plt.tight_layout()
# plt.show()

######################################
# from tacotron.utils.text import line_split
# line = "there are four major principles that nearly all great investors use to guide them in making investment decisions. I call these the Core Four. CORE PRINCIPLE 1: DON'T LOSE, CORE PRINCIPLE 2: ASYMMETRIC RISK/REWARD, CORE PRINCIPLE 3: TAX EFFICIENCY, CORE PRINCIPLE 4: DIVERSIFICATION"
# line_split(line)

#######################################

# folder = '/raid1/stephen/rayhane-tc2/Tacotron-2/record_wav_lastnight'
#
# files = os.listdir(folder)
#
# files = [file for file in files if file.endswith('.wav')]
#
# for name in tqdm(files):
#     new_name = name.replace('.wav', '.mp3')
#     cmd = f'cp {os.path.join(folder, name)} {os.path.join(folder, new_name)}'
#     os.system(cmd)


##########################################
# wav_folder = '/raid1/stephen/rayhane-tc2/Tacotron-2/record_wav_lastnight'
def foo():
    class exception_1(Exception): pass
    try:
        raise exception_1()
    except exception_1:
        print('haha')

foo()
