from pathlib import Path

import wave
import contextlib

def read_duration(fname):
	with contextlib.closing(wave.open(fname, 'r')) as f:
		frames = f.getnframes()
		rate = f.getframerate()
		duration = frames / float(rate)
	return duration


if __name__ == '__main__':
	record_txt_path = Path('..','new_record_txt_line')
	record_wav_path = Path('..','record_wav')


	txt_list = sorted(record_txt_path.glob('*'))

	txt_wav_list = []


	len_txt_list = [len(open(file.resolve(),'r').readline()) for file in txt_list]
	duration_wav_list = [read_duration(f'{record_wav_path.resolve()}/{file.name}.mp3') for file in txt_list]

	ratio_list = [dur/len for len, dur in zip(len_txt_list,duration_wav_list)]

	name_ratio_list = [(name, ratio)for name, ratio in zip(txt_list, ratio_list)]
	ratio_average = sum(ratio_list)/len(ratio_list)
	name_ratio_list_small = sorted(name_ratio_list, key=lambda x:x[1])
	name_ratio_list_big = sorted(name_ratio_list, key=lambda x:x[1], reverse=True)
	a = 1

