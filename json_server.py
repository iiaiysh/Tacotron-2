from flask import Flask, request, Response
from flask_cors import CORS
from flask import render_template, jsonify, send_file
import os
import io
import re

import argparse
from hparams import hparams, hparams_debug_string
from hparams_ysh import hparams_ysh, update_hp1_with_hp2

from tacotron.synthesizer_ysh import Synthesizer



def precess2syninput(text):
    replace_marks = ',:-;.?!'
    for mark in replace_marks:
        text = text.replace(mark, mark + ' ~ ')
    text = ' ~ ' + text
    return text


def line_split(text):
    split_marks = '.!?'
    text_list = re.split(f'([{split_marks}])', text)
    text_list_without_standalone_mark = []
    for i in range(0, len(text_list), 2):
        if i + 1 < len(text_list):
            text_list_without_standalone_mark.append(text_list[i] + text_list[i + 1])
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






# class AttrDict(dict):
#     def __init__(self, *args, **kwargs):
#         super(AttrDict, self).__init__(*args, **kwargs)
#         self.__dict__ = self
#
#
# args = AttrDict()
# args.update({
#                 'checkpoint': '/raid1/stephen/rayhane-tc2-noencoder-demo/logs-Tacotron-2/taco_pretrained/tacotron_model.ckpt-50000',
#                 'port': 8080})


# @app.route("/", methods=['GET'])
# def front_end():
#    return render_template('upload_file_json.html')
app = Flask(__name__)
CORS(app)

@app.route('/test')
def test():
    return "hello world!"


@app.route('/demo')
def demo():
    return render_template('demo.html')


@app.route('/synthesize', methods=['GET', 'POST'])
def synthesize_result():
    text = request.values.get('text')
    filename = synthesizer.synthesize(text,None,None,None,None,return_wav = True)

    # return send_file(io.BytesIO(data), mimetype='audio/wav')
    return send_file(filename, mimetype='audio/wav')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Full path to model checkpoint')
    parser.add_argument('--port',default=8080)
    parser.add_argument('--hparams', default='', help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    args = parser.parse_args()

    hparams.parse(args.hparams)
    update_hp1_with_hp2(hparams, hparams_ysh)
    hparams.set_hparam('tacotron_num_gpus',1)
    hparams.set_hparam('tacotron_synthesis_batch_size',1)

    synthesizer = Synthesizer()
    synthesizer.load(args.checkpoint, hparams)

    print('Serving on port %d' % args.port)

    app.run(host="0.0.0.0", port=args.port)




