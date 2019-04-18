import argparse
import os
import re
from hparams import hparams, hparams_debug_string
from tacotron.synthesizer_simple import Synthesizer
import pprint
from tqdm import tqdm
import time
import numpy as np
import logging
print(__file__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler(f"{__file__.split('/')[-1].split('.')[0]}.log"),
        logging.StreamHandler()
    ])
logger = logging.getLogger()

def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


def run_eval(args):
  print(hparams_debug_string())


replace_marks = ',:-;.?!'
def precess2syninput(text):
  for mark in replace_marks:
    text = text.replace(mark,mark+' ~ ')
  text = ' ~ ' + text
  return text

def line_split(text):
  split_marks = '.!?'
  text_list = re.split(f'([{split_marks}])',text)
  text_list_without_standalone_mark = []
  for i in range(0,len(text_list),2):
    if i+1 < len(text_list):
      text_list_without_standalone_mark.append(text_list[i] + text_list[i+1])
    else:
      text_list_without_standalone_mark.append(text_list[i])

  sentence_list = []
  old_sentence = ''
  for item in text_list_without_standalone_mark:
    new_sentence = old_sentence + item
    if len(new_sentence.split(' ')) > 20:
      sentence_list.append(old_sentence)
      old_sentence = item
    else:
      old_sentence = new_sentence
  
  if old_sentence != '':
    sentence_list.append(old_sentence)
  return sentence_list

def walkdir(folder):
    """Walk through each files in a directory"""
    for dirpath, dirs, files in os.walk(folder):
        for filename in files:
            yield os.path.abspath(os.path.join(dirpath, filename))

if __name__ == '__main__':
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
  args = parser.parse_args()

  #checkpoint = '/home/shihangyu/logs-audiobook_manual_finetune_ljgeek_38000/model.ckpt-48800'
  synth = Synthesizer()
  synth.load(args.checkpoint,hparams)

  base_path = get_output_base_path(args.checkpoint)


  record_txt_path = '../record_txt'
  record_wav_path = '../record_wav'
  #new_record_wav_path = 'new_record_wav'
  num_file = 0
  os.makedirs(record_wav_path,exist_ok=True)

  file_path_list = []
  for dirpath, dirs, files in os.walk(record_txt_path):
    for file in files:
      if not (file=='.DS_Store' or file.endswith('~') or file.startswith('~') or file.startswith('_')):
        file_path = os.path.join(dirpath,file)
        file_path_list.append(file_path)

  t1 = time.time()
  total_wav_counter = 0
  for file_path in tqdm(file_path_list,position=0):
    try:
      with open(file_path,'r',encoding = 'utf-8') as f:
        lines = f.readlines()

        index_wav = 1
        flag_fistline = True
        logger.debug(f'synthesize in the file...{file_path}')
        for num_line, line in enumerate(lines):
          if num_line == 0:
            continue

          if re.match('\#+',line.strip()):
            # print(line)
            continue
          
          m = re.match('CONTEXT:(.*)\d',line.strip())
          if m:
            line = m.group(1)

          
          line = line.strip()
          line = line.replace('*','')
          line = line.replace('-','')

          #sentence_list = line_split(line)
          #syn_input_list = [precess2syninput(sentence) for sentence in sentence_list]
          # pprint.pprint(syn_input_list)
          if line is '':
              continue
          outwav,mels = synth.synthesize(line)

          save_name = '-'.join(file_path.split('/')[1:])
          path = f'{record_wav_path}/{save_name}-{num_line+1}.wav'
          #path = f'{record_wav_path}/{save_name}-{index_wav}.wav'
          #new_path = f'{new_record_wav_path}/{save_name}-{num_line+1}.wav'
          #os.system(f'mv {path} {new_path}')
          for si,mel in enumerate(mels):
              mel_path = f'{record_wav_path}/{save_name}-{num_line+1}-split-{si}.npy'
              np.save(mel_path, mel, allow_pickle=False)
          logger.debug('Synthesizing: %s' % path)  
          total_wav_counter += 1 
          with open(path, 'wb') as fout:
              index_wav += 1
              fout.write(outwav)
    except Exception as e:
      logger.error(file_path)
      logger.error(str(e))
      raise Exception(str(e))
      

