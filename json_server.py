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
from tacotron.utils.text import line_split

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
    print('synthesize...')
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
    return send_file(os.path.join(log_dir, f'wavs/wav-{wav_name}-linear.mp3'), mimetype='audio/mp3')


@app.route('/synthesize', methods=['GET', 'POST'])
def synthesize():
    print('synthesize split...')
    text = request.values.get('text')
    # text.replace(',', ':')
    # wav_name = text.split(' ')[0]
    wav_name = 'tmp'
    text_split = line_split(text)

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
    for i_part, part_text in enumerate(text_split):
        texts = [part_text]
        wav_name_part = f'{wav_name}-p{i_part}'
        wav_name_part_list.append(wav_name_part)

        # # return send_file(io.BytesIO(data), mimetype='audio/wav')

        synthesizer.synthesize(texts, [f'{wav_name_part}'], eval_dir, log_dir, ['/raid1/stephen/rayhane-tc2/Tacotron-2/training_data/blizzard_training_data_maxmel_1700_clip_norescale/mels/mel-TheManThatCorruptedHadleyburg_chp37_00020.npy'], return_wav=True, wav_format='mp3')



    #concate parts into whole wav
    whole_path = os.path.join(log_dir, f'wavs/wav-{wav_name}-linear.mp3')
    if len(wav_name_part_list) == 1:
        part0_path = os.path.join(log_dir, f'wavs/wav-{wav_name_part_list[0]}-linear.mp3')
        os.system(f'cp {part0_path} {whole_path}')

    elif len(wav_name_part_list) > 1:
        wav_concate = None
        for wav_part in wav_name_part_list:
            part_path = os.path.join(log_dir, f'wavs/wav-{wav_part}-linear.mp3')
            wav_tmp = load_wav(part_path, sr=hparams.sample_rate)
            if wav_concate is None:
                wav_concate = wav_tmp
            else:
                wav_concate = np.concatenate([wav_concate, wav_tmp])
        save_wav(wav_concate, whole_path, hparams.sample_rate)
    else:
        raise RuntimeError('why wav_parts have no items')

    # return send_file(io.BytesIO(data), mimetype='audio/wav')
    return send_file(whole_path, mimetype='audio/mp3')


@app.route('/retrieve', methods=['GET', 'POST'])
def retrieve():
    text = request.values.get('text')


    if not os.path.exists(text):
        return "Can not find file, please check the input path"
    else:
    # return send_file(io.BytesIO(data), mimetype='audio/wav')
        return send_file(text, mimetype='audio/mp3')

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




