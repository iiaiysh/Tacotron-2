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
# line = '''"Let me tell you one story. Bill Gates has taken big risks. He was at Harvard, had a great education setup. But then he read a magazine article about the first personal computer called the Altair and said, “You know what, I think this is the future.” He anticipated the changes that would come in the future. He had guts and he took a risk. He called the Altair guy up and told him he’d written software that would make the computer run perfectly. He totally made it up. He was just being gutsy and figured “If I tell the guy I got software and he says come see me, I'm gonna figure out how to do it.” The guy said to bring it in three weeks, so he had three weeks to come up with software. He and his partner worked around the clock. And, literally, when he went to Altair in New Mexico to show this software, he didn't have a computer to test it on, he didn't know it was going to work for sure. Thank God for his future it worked. It was a gutsy risky thing; he invested all this time and energy, he blew off his school, he did everything. And thank God for him it worked. But that didn't make him wealthy. What made him wealthy was something called MSDOS. MSDOS was a software that he did not write. He bought MSDOS, which made him three and a half billion dollars, for $50,000. Bill Gates bought it for $50,000 from somebody else. He saw IBM and he studied his competition and he said “They think they're in the computer business and I think the computer business is going to become a commodity at some point in the future. I want to control the intelligence that controls those computers.” He thought “I'll offer IBM the software and I'll just go buy it from this kid over here.” $50 000 is a huge sum of money. Most business people don't have guts, they don't take risk and they don't invest. He went to IBM and convinced them.  Now, IBM wanted it, but they wanted to buy it from him. He said "No, I'll rent it to you." They said no. But he said, “Look, you're not in the software business anyway, you're in the computer business. You don't want software, software changes. You don't know that area, let me do this for you.”You have to anticipate shifts. The solution is constant and neverending improvement in the areas that control the success or failure of your business. Constant, neverending improvement. It doesn’t have to be anything huge but it has to be constant, incremental improvements in the areas that matter on a regular basis. You will dominate as long as you also anticipate.'''
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
# def foo():
#     class exception_1(Exception): pass
#     try:
#         raise exception_1()
#     except exception_1:

#         print('haha')
#
# foo()

def dec(func):

    print('this is dec')
    return func

@dec
def moo(text: str, number: int) -> str:
    print(text, number)
    return 1



if  __name__ == '__main__':
    a = 1
    # moo(text=1, number='test')
