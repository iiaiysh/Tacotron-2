import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument('--folder', required=True, help='Path to model checkpoint')
parser.add_argument('--name', required=True, help='Path to model checkpoint')
args = parser.parse_args()

ckpt_list = os.listdir(args.folder)
ckpt_list = [ckpt[:-5] for ckpt in ckpt_list if ckpt.endswith('.meta')]

output_dir = os.path.join('tacotron_output',f'output_{args.name}')

wav_dir = os.path.join(output_dir,'logs-eval','wavs')

wav_list = []
if os.path.exists(wav_dir):
	wav_list = os.listdir(wav_dir)
	wav_list = [wav for wav in wav_list if wav.endswith('mel.wav')]


for ckpt in ckpt_list:
	m = re.match('tacotron_model.ckpt-(\d+)', ckpt)
	ckpt_step = m.group(1)

	if f'wav-step_{ckpt_step}_batch_0_sentence_0-mel.wav' in wav_list:
		print(f'ckpt {ckpt} already synthesized in output_{args.name} folder, continue...')
		continue
	cmd = f'python synthesize_ysh.py --checkpoint {os.path.join(args.folder,ckpt)} --name {args.name}'
	os.system(cmd)