import datasets.audio as audio
import numpy as np
from hparams import hparams

# blizzard
# name_list = [
# 'training_data/blizzard_training_data_new_maxmel_1700/audio/audio-ATrampAbroad_chp01_00002.npy',
# 'training_data/blizzard_training_data_new_maxmel_1700/audio/audio-ATrampAbroad_chp01_00003.npy',
# 'training_data/blizzard_training_data_new_maxmel_1700/audio/audio-ATrampAbroad_chp01_00005.npy',
# 'training_data/blizzard_training_data_new_maxmel_1700/audio/audio-ATrampAbroad_chp01_00006.npy',
#
# 'training_data/blizzard_training_data_new_maxmel_1700/mels/mel-ATrampAbroad_chp01_00002.npy',
# 'training_data/blizzard_training_data_new_maxmel_1700/mels/mel-ATrampAbroad_chp01_00003.npy',
# 'training_data/blizzard_training_data_new_maxmel_1700/mels/mel-ATrampAbroad_chp01_00005.npy',
# 'training_data/blizzard_training_data_new_maxmel_1700/mels/mel-ATrampAbroad_chp01_00006.npy',
#
# 'training_data/blizzard_training_data_new_maxmel_1700/linear/linear-ATrampAbroad_chp01_00002.npy',
# 'training_data/blizzard_training_data_new_maxmel_1700/linear/linear-ATrampAbroad_chp01_00003.npy',
# 'training_data/blizzard_training_data_new_maxmel_1700/linear/linear-ATrampAbroad_chp01_00005.npy',
# 'training_data/blizzard_training_data_new_maxmel_1700/linear/linear-ATrampAbroad_chp01_00006.npy',
# ]

# ljspeech
# name_list = [
# '/raid1/stephen/rayhane-tc2-noencoder-demo/ljspeech-training_data/linear/linear-LJ001-0001.npy',
# '/raid1/stephen/rayhane-tc2-noencoder-demo/ljspeech-training_data/linear/linear-LJ001-0002.npy',
# '/raid1/stephen/rayhane-tc2-noencoder-demo/ljspeech-training_data/linear/linear-LJ001-0003.npy',
# '/raid1/stephen/rayhane-tc2-noencoder-demo/ljspeech-training_data/linear/linear-LJ001-0004.npy',
# '/raid1/stephen/rayhane-tc2-noencoder-demo/ljspeech-training_data/mels/mel-LJ001-0001.npy',
# '/raid1/stephen/rayhane-tc2-noencoder-demo/ljspeech-training_data/mels/mel-LJ001-0002.npy',
# '/raid1/stephen/rayhane-tc2-noencoder-demo/ljspeech-training_data/mels/mel-LJ001-0003.npy',
# '/raid1/stephen/rayhane-tc2-noencoder-demo/ljspeech-training_data/mels/mel-LJ001-0004.npy',
# '/raid1/stephen/rayhane-tc2-noencoder-demo/ljspeech-training_data/audio/audio-LJ001-0001.npy',
# '/raid1/stephen/rayhane-tc2-noencoder-demo/ljspeech-training_data/audio/audio-LJ001-0002.npy',
# '/raid1/stephen/rayhane-tc2-noencoder-demo/ljspeech-training_data/audio/audio-LJ001-0003.npy',
# '/raid1/stephen/rayhane-tc2-noencoder-demo/ljspeech-training_data/audio/audio-LJ001-0004.npy',
# ]

# new anno
name_list = [
'training_data/new_annotation_training_data_maxmel_1500_clip/audio/audio-ue98-98817-107993.npy',
'training_data/new_annotation_training_data_maxmel_1500_clip/audio/audio-ue98-95467-98653.npy',
'training_data/new_annotation_training_data_maxmel_1500_clip/audio/audio-ue98-79777-95296.npy',


'training_data/new_annotation_training_data_maxmel_1500_clip/mels/mel-ue98-98817-107993.npy',
'training_data/new_annotation_training_data_maxmel_1500_clip/mels/mel-ue98-95467-98653.npy',
'training_data/new_annotation_training_data_maxmel_1500_clip/mels/mel-ue98-79777-95296.npy',

'training_data/new_annotation_training_data_maxmel_1500_clip/linear/linear-ue98-98817-107993.npy',
'training_data/new_annotation_training_data_maxmel_1500_clip/linear/linear-ue98-95467-98653.npy',
'training_data/new_annotation_training_data_maxmel_1500_clip/linear/linear-ue98-79777-95296.npy',
]

for name in name_list:
	type = name.split('/')[-2]
	savename = name.split('/')[-1][:-4]

	if type == 'audio':
		wav = np.load(name)
		audio.save_wav(wav, f'{savename}.wav', 22050)

	elif type == 'mels':
		mel = np.load(name)
		wav = audio.inv_mel_spectrogram(mel.T, hparams)
		audio.save_wav(wav, f'{savename}.wav', 22050)

	elif type == 'linear':
		li = np.load(name)
		wav = audio.inv_linear_spectrogram(li.T, hparams)
		audio.save_wav(wav, f'{savename}.wav', 22050)
	else:
		raise RuntimeError('unknown type')

