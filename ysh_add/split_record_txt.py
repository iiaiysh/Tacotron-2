import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import re


rootdir = os.path.join(os.path.realpath(__file__), os.pardir)
read_folder = 'record_txt'
save_folder = 'new_record_txt_line'
os.makedirs(save_folder, exist_ok=True)

for dirpath, dirs, files in os.walk(read_folder):
    for file in files:
        if not (file == '.DS_Store' or file.endswith('~') or file.startswith('~') or file.startswith('_')):
            relpath = os.path.relpath(dirpath, read_folder)
            name_file = relpath.replace('/','-') + '-' + file

            lines = open(os.path.join(dirpath, file)).readlines()

            for num_line, line in enumerate(lines, start=1):
                if num_line == 1:
                    continue

                if re.match('\#+', line.strip()):
                    continue

                m = re.match('CONTEXT:(.*)\d', line.strip())
                if m:
                    line = m.group(1)

                line = line.strip()
                line = line.replace('*', '')
                line = line.replace('-', '')

                # sentence_list = line_split(line)
                # syn_input_list = [precess2syninput(sentence) for sentence in sentence_list]
                # pprint.pprint(syn_input_list)
                if line is '':
                    continue

                name_line = name_file+'-'+str(num_line)

                fw = open(os.path.join(save_folder, name_line), 'w')
                fw.write(line)

            # os.system(f'cp {os.path.join(dirpath, file)} {os.path.join(save_folder, save_name)}')
