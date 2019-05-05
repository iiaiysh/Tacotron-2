from flask import Flask, request, Response
from flask_cors import CORS
from flask import render_template, jsonify, send_file
import os
import io
import re
import numpy as np

import argparse
from hparams import hparams, hparams_debug_string
from hparams_ysh import hparams_ysh, update_hp1_with_hp2

from tacotron.synthesizer_ysh import Synthesizer
from tacotron.synthesizer_ysh_split_init_load import Synthesizer_Split
from datasets.audio import load_wav, save_wav
from tacotron.utils.text import line_split, line_split_at

import uuid

import time

def precess2syninput(text):
    replace_marks = ',:-;.?!'
    for mark in replace_marks:
        text = text.replace(mark, mark + ' ~ ')
    text = ' ~ ' + text
    return text


app = Flask(__name__)
CORS(app)

@app.route('/test')
def test():
    return "hello world! this is tony voice!"


@app.route('/demo')
def demo():
    return render_template('demo.html')

@app.route('/synthesize_no_split', methods=['GET', 'POST'])
def synthesize_no_split():
    print('\n')

    t1 = time.time()

    text = request.values.get('text')
    # text.replace(',', ':')
    # wav_name = text.split(' ')[0]
    wav_name = 'tmp'

    texts = [text]
    # filename = synthesizer.synthesize(text,None,None,None,None,return_wav = True)
    output_dir = os.path.join('tacotron_output', 'output_tmp')
    eval_dir = os.path.join(output_dir, 'eval')
    log_dir = os.path.join(output_dir, 'logs-eval')

    # Create output path if it doesn't exist
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'wavs'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'plots-linear'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'plots-mel'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'plots-align'), exist_ok=True)

    # mel_filenames, speaker_ids = synthesizer.synthesize(texts, ['tmp'], eval_dir, log_dir, ['/raid1/stephen/rayhane-tc2/Tacotron-2/training_data/blizzard_training_data_maxmel_1700_clip_norescale/mels/mel-TheManThatCorruptedHadleyburg_chp37_00020.npy'], return_wav=True, wav_format='mp3')
    # # return send_file(io.BytesIO(data), mimetype='audio/wav')
    # return send_file(os.path.join(log_dir, 'wavs/wav-tmp-linear.mp3'), mimetype='audio/mp3')

    mel_filenames, speaker_ids = synthesizer.synthesize(texts, [f'{wav_name}'], eval_dir, log_dir, ['/raid1/stephen/rayhane-tc2/Tacotron-2/training_data/blizzard_training_data_maxmel_1700_clip_norescale/mels/mel-TheManThatCorruptedHadleyburg_chp37_00020.npy'], return_wav=True, wav_format='mp3')
    # return send_file(io.BytesIO(data), mimetype='audio/wav')

    print('synthesize time...{}', time.time()-t1)

    return send_file(os.path.join(log_dir, f'wavs/wav-{wav_name}-linear.mp3'), mimetype='audio/mp3')


@app.route('/synthesize', methods=['GET', 'POST'])
def synthesize():
    print('\n')
    t1 = time.time()

    text = request.values.get('text').strip()
    assert type(text) == str
    assert text != ''

    delete_uuid_part = True

    wav_name = f'tmp-{str(uuid.uuid1())}'
    text_split = line_split_at(text)

    output_dir = os.path.join('tacotron_output', 'output_tmp')
    eval_dir = os.path.join(output_dir, 'eval')
    log_dir = os.path.join(output_dir, 'logs-eval')

    # Create output path if it doesn't exist
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'wavs'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'plots-linear'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'plots-mel'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'plots-align'), exist_ok=True)

    wav_name_part_list = []
    wav_path_list = []
    for i_part, part_text in enumerate(text_split):
        texts = [part_text]
        wav_name_part = f'{wav_name}-p{i_part}'
        wav_name_part_list.append(wav_name_part)

        # # return send_file(io.BytesIO(data), mimetype='audio/wav')
        for i_repeat in range(5):
            wav_path, wav_duration = synthesizer.synthesize(texts, [f'{wav_name_part}'], eval_dir, log_dir, ['/raid1/stephen/rayhane-tc2/Tacotron-2/training_data/blizzard_training_data_maxmel_1700_clip_norescale/mels/mel-TheManThatCorruptedHadleyburg_chp37_00020.npy'], return_wav=True, wav_format='mp3')
            char_ratio = wav_duration / len(part_text)
            word_ratio = wav_duration / len(part_text.split(' '))
            print(f'         [char:{char_ratio:.4f}] [word:{word_ratio:.4f}]', wav_path)
            # except only one word dont need to repeat
            if (char_ratio < 0.1 and word_ratio < 0.8) or (len(part_text.split(' '))==2):
                break
        wav_path_list.append(wav_path)

    # use wav_name_part_list to concatenate
    # #concate parts into whole wav
    # whole_path = os.path.join(log_dir, f'wavs/wav-{wav_name}-linear.mp3')
    # if len(wav_name_part_list) == 1:
    #     part0_path = os.path.join(log_dir, f'wavs/wav-{wav_name_part_list[0]}-linear.mp3')
    #     os.system(f'cp {part0_path} {whole_path}')
    #
    # elif len(wav_name_part_list) > 1:
    #     wav_concate = None
    #     for wav_part in wav_name_part_list:
    #         part_path = os.path.join(log_dir, f'wavs/wav-{wav_part}-linear.mp3')
    #         wav_tmp = load_wav(part_path, sr=hparams.sample_rate)
    #         if wav_concate is None:
    #             wav_concate = wav_tmp
    #         else:
    #             wav_concate = np.concatenate([wav_concate, wav_tmp])
    #     save_wav(wav_concate, whole_path, hparams.sample_rate)
    # else:
    #     raise RuntimeError('why wav_parts have no items')


    # use wav_path_list to concatenate
    #concate parts into whole wav
    whole_path = os.path.join(log_dir, f'wavs/wav-{wav_name}-linear.mp3')

    if len(wav_path_list) == 1:
        part0_path = wav_path_list[0]
        os.system(f'cp {part0_path} {whole_path}')

    elif len(wav_path_list) > 1:
        wav_concate = None
        for wav_path in wav_path_list:
            wav_tmp = load_wav(wav_path, sr=hparams.sample_rate)
            if wav_concate is None:
                wav_concate = wav_tmp
            else:
                both_max_value = max(np.max(np.abs(wav_tmp)), np.max(np.abs(wav_tmp)))
                wav_concate = np.concatenate([wav_concate/np.max(np.abs(wav_concate))*both_max_value, wav_tmp/np.max(np.abs(wav_tmp))*both_max_value])
        save_wav(wav_concate, whole_path, hparams.sample_rate)
    else:
        raise RuntimeError('why wav_parts have no items')

    if delete_uuid_part:
        for wav_path in wav_path_list:
            try:
                os.system(f'rm {wav_path}')
            except:
                pass

    # return send_file(io.BytesIO(data), mimetype='audio/wav')
    print('synthesize split time...{}', time.time()-t1)

    return send_file(whole_path, mimetype='audio/mp3')


@app.route('/retrieve', methods=['GET', 'POST'])
def retrieve():
    print('\n')

    t1 = time.time()

    text = request.values.get('text').strip()
    assert type(text) == str
    assert text != ''
    surfix = 'mp3'

    fullname  = f'{text}.{surfix}'
    # dirpath = '/raid1/mo/tony_robbins/video_root/speech_0422'
    dirpath = '/raid1/stephen/rayhane-tc2/Tacotron-2/record_wav'

    print('retrieve time...{}', time.time()-t1)

    if not os.path.exists(os.path.join(dirpath, fullname)):
        return "Can not find file, please check the input path"
    else:
    # return send_file(io.BytesIO(data), mimetype='audio/wav')
        return send_file(os.path.join(dirpath, fullname), mimetype='audio/mp3')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='logs-rayhane-Tacotron/taco_pretrained/blizzard_pretrain_0217/tacotron_model.ckpt-158000', help='Full path to model checkpoint')
    parser.add_argument('--port', default=5001)
    parser.add_argument('--hparams', default='', help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    args = parser.parse_args()

    update_hp1_with_hp2(hparams, hparams_ysh)
    hparams.parse(args.hparams)
    hparams.set_hparam('tacotron_num_gpus',1)
    hparams.set_hparam('tacotron_synthesis_batch_size',1)
    hparams.set_hparam('cleaners','ysh_cleaners')

    synthesizer = Synthesizer_Split(hparams)

    synthesizer.load(args.checkpoint)

    print(f'Serving on port {args.port}')

    app.run(host="0.0.0.0", port=args.port)




