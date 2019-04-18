import datasets.audio as audio
import numpy as np
from hparams import hparams
import os
import argparse

paser = argparse.ArgumentParser()
paser.add_argument('--root_path', default='/raid1/stephen/rayhane-tc2/Tacotron-2/training_data/newanno0307_maxmel1500_norescale_fmax7600_clip_duration_ratio')
args = paser.parse_args()


traingingdata_path = args.root_path
root, dirs, files = os.walk(traingingdata_path).__next__()

paths_list = []
savepaths_list = []
for dir in ['linear']:# change this if your training data has different folders
	names = os.listdir(os.path.join(traingingdata_path, dir))
	names = [name for name in names if name.endswith('.npy')]

	paths = [os.path.join(traingingdata_path, dir, name) for name in names]
	paths_list.extend(paths)

	savedir = f'{dir}-wavs'
	os.makedirs(os.path.join(traingingdata_path, savedir), exist_ok=True)
	savepaths = [os.path.join(traingingdata_path, savedir, name.replace('npy', 'wav')) for name in names]
	savepaths_list.extend(savepaths)

assert len(paths_list) == len(savepaths_list)
for i, path in enumerate(paths_list):
	type = path.split('/')[-2]

	savepath = savepaths_list[i]

	if os.path.exists(savepath):
		continue

	if type == 'audio':
		wav = np.load(path)
		audio.save_wav(wav, savepath, 22050)

	elif type == 'mels':
		mel = np.load(path)
		wav = audio.inv_mel_spectrogram(mel.T, hparams)
		audio.save_wav(wav, savepath, 22050)

	elif type == 'linear':
		li = np.load(path)
		wav = audio.inv_linear_spectrogram(li.T, hparams)
		audio.save_wav(wav, savepath, 22050)
	else:
		raise RuntimeError('unknown type')

	print(f'process [{i}]/[{len(paths_list)}] samples')

