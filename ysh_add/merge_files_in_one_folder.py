from datasets.audio import load_wav, save_wav
import os
import numpy as np

postfix = 'mp3'
read_folder = 'need_merge'
sr = 22050

file_list = os.listdir(read_folder)
file_list = [file for file in file_list if file.endswith(postfix)]

assert len(file_list) >= 1
file_list_sort = sorted(file_list, key = lambda x: int(x.split('.')[0]))

wav_concate = None
for file in file_list_sort:
    wav_tmp = load_wav(os.path.join(read_folder, file), sr)
    if wav_concate is None:
        wav_concate = wav_tmp
    else:
        wav_concate = np.concatenate([wav_concate, wav_tmp])

save_path = os.path.join(read_folder,'merge.mp3')
save_wav(wav_concate, save_path, sr)