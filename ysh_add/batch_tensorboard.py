import os
import argparse
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument('--logname', choices=['tacotron','wavenet'], required=True)

args = parser.parse_args()


root_path = 'logs'


for root, dirs, files in os.walk(root_path,):
	print(root)
	print(dirs)
	print(files)
	break

subprocess.run(["conda ", "activate", "rayhane-ysh"])


port = 6006
for dir in dirs:
	if not dir.startswith('logs-'):
		continue

	if not os.path.exists(os.path.join(root,dir,f'{args.logname}_events')):
		continue

	event_dir = os.path.join(root,dir,f'{args.logname}_events')

	try:
		os.system(f'tensorboard --logdir {event_dir}')
	except Exception as e:
		a = 1
