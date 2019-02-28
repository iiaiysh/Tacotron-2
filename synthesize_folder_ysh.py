import argparse
import os
import re
from hparams_ysh import hparams_ysh

parser = argparse.ArgumentParser()
parser.add_argument('--folder', required=True, help='Path to model checkpoint')
parser.add_argument('--name', required=True, help='Path to model checkpoint')
args = parser.parse_args()

ckpt_list = os.listdir(args.folder)
ckpt_list = [ckpt[:-5] for ckpt in ckpt_list if ckpt.endswith('.meta')]

output_dir = os.path.join('tacotron_output',f'output_{args.name}')

wav_dir = os.path.join(output_dir,'logs-eval','wavs')

wav_count_dict = {}
if os.path.exists(wav_dir):
	wav_list = os.listdir(wav_dir)
	wav_list = [wav.split('batch')[0] for wav in wav_list if wav.endswith('.wav')]

	wav_set = set(wav_list)
	wav_count_dict_list = [{wav: wav_list.count(wav)} for wav in wav_set]
	[wav_count_dict.update(_) for _ in wav_count_dict_list]

for ckpt in ckpt_list:
	m = re.match('tacotron_model.ckpt-(\d+)', ckpt)
	ckpt_step = m.group(1)

	if f'wav-step_{ckpt_step}_' in wav_count_dict.keys():
		if wav_count_dict[f'wav-step_{ckpt_step}_'] == len(hparams_ysh.sentences)*2:
			do_synthesize = False
		else:
			do_synthesize = True
	else:
		do_synthesize = True

	if do_synthesize:
		cmd = f'python synthesize_ysh.py --checkpoint {os.path.join(args.folder, ckpt)} --name {args.name}'
		os.system(cmd)
	else:
		print(f'ckpt {ckpt} already synthesized in output_{args.name} folder, continue...')
		continue
