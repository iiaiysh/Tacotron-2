import os
import wave
import contextlib
import sys
import numpy as np
import pickle
def read_duration(fname):
        with contextlib.closing(wave.open(fname,'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
        return duration


# mel_path = 'training_data/new_annotation_training_data_maxmel_1500/mels/'
# audio_path = 'training_data/new_annotation_training_data_maxmel_1500/audio/'
# linear_path = 'training_data/new_annotation_training_data_maxmel_1500/linear/'
#
# mel_list = os.listdir(mel_path)
# audio_list = os.listdir(audio_path)
# linear_list = os.listdir(linear_path)
#
# mel_list = [np.load(os.path.join(mel_path, mel)) for mel in mel_list]
# audio_list = [np.load(os.path.join(audio_path, audio)) for audio in audio_list]
# linear_list = [np.load(os.path.join(linear_path, linear)) for linear in linear_list]
f = open('relation_list.file','rb')

relation_list = pickle.load(f)

mel = np.load('training_data/blizzard_training_data_maxmel_1700_clip_norescale/mels/mel-TheManThatCorruptedHadleyburg_chp37_00019.npy')


mel = np.load('tacotron_output/output_blizzard_pretrain_sym69_6gpu_norescale_ckpt48000_wavenet_train/gta/mel-TheManThatCorruptedHadleyburg_chp37_00019.npy')

mel = np.load('training_data/blizzard_training_data_maxmel_1700_clip_norescale/mels/mel-TheManThatCorruptedHadleyburg_chp37_00019.npy')
wav = audio.inv_mel_spectrogram(mel.T, hparams)
audio.save_wav(wav,'gta-37_00019.wav',22050)