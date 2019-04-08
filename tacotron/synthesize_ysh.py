import argparse
import os
import re
import time
from time import sleep

import tensorflow as tf
from hparams import hparams, hparams_debug_string
from infolog import log
from tacotron.synthesizer_ysh_split_init_load import Synthesizer_Split
from tacotron.synthesizer_ysh import Synthesizer

from tqdm import tqdm


def generate_fast(model, text):
    model.synthesize([text], None, None, None, None)


def run_live(args, checkpoint_path, hparams):
    #Log to Terminal without keeping any records in files
    log(hparams_debug_string())
    synth = Synthesizer()
    synth.load(checkpoint_path, hparams)

    #Generate fast greeting message
    greetings = 'Hello, Welcome to the Live testing tool. Please type a message and I will try to read it!'
    log(greetings)
    generate_fast(synth, greetings)

    #Interaction loop
    while True:
        try:
            text = input()
            generate_fast(synth, text)

        except KeyboardInterrupt:
            leave = 'Thank you for testing our features. see you soon.'
            log(leave)
            generate_fast(synth, leave)
            sleep(2)
            break

def run_eval(args, checkpoint_path, output_dir, hparams, sentences):
    eval_dir = os.path.join(output_dir, 'eval')
    log_dir = os.path.join(output_dir, 'logs-eval')

    if args.model == 'Tacotron-2':
        assert os.path.normpath(eval_dir) == os.path.normpath(args.mels_dir)

    #Create output path if it doesn't exist
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'wavs'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'plots-linear'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'plots-mel'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'plots-align'), exist_ok=True)

    log(hparams_debug_string())
    synth = Synthesizer()
    synth.load(checkpoint_path, hparams)

    #Set inputs batch wise
    sentences = [sentences[i: i+hparams.tacotron_synthesis_batch_size] for i in range(0, len(sentences), hparams.tacotron_synthesis_batch_size)]

    m = re.match('(.*)/tacotron_model.ckpt-(\d+)', checkpoint_path)
    ckpt_step = m.group(2)

    log('Starting Synthesis')
    with open(os.path.join(eval_dir, 'map.txt'), 'w') as file:
        for i, texts in enumerate(tqdm(sentences)):
            start = time.time()
            basenames = ['step_{}_batch_{}_sentence_{}'.format(ckpt_step, i, j) for j in range(len(texts))]
            mel_filenames, speaker_ids = synth.synthesize(texts, basenames, eval_dir, log_dir, None)

            for elems in zip(texts, mel_filenames, speaker_ids):
                file.write('|'.join([str(x) for x in elems]) + '\n')
    log('synthesized mel spectrograms at {}'.format(eval_dir))
    return eval_dir

def run_eval_experiment(args, checkpoint_path, output_dir, hparams, sentences):
    # update_vars = [v for v in self.all_vars if
    #                not ('inputs_embedding' in v.name or 'encoder_' in v.name)] if hp.tacotron_fine_tuning else None
    eval_dir = os.path.join(output_dir, 'eval')
    log_dir = os.path.join(output_dir, 'logs-eval')

    if args.model == 'Tacotron-2':
        assert os.path.normpath(eval_dir) == os.path.normpath(args.mels_dir)

    #Create output path if it doesn't exist
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'wavs'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'plots-linear'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'plots-mel'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'plots-align'), exist_ok=True)

    log(hparams_debug_string())
    synth = Synthesizer()
    synth.load(checkpoint_path, hparams, is_evaluating=True)

    #Set inputs batch wise
    sentences = [sentences[i: i+hparams.tacotron_synthesis_batch_size] for i in range(0, len(sentences), hparams.tacotron_synthesis_batch_size)]

    m = re.match('(.*)/tacotron_model.ckpt-(\d+)', checkpoint_path)
    ckpt_step = m.group(2)

    log('Starting Synthesis')
    with open(os.path.join(eval_dir, 'map.txt'), 'w') as file:
        for i, texts in enumerate(tqdm(sentences)):
            start = time.time()
            basenames = ['step_{}_batch_{}_sentence_{}'.format(ckpt_step, i, j) for j in range(len(texts))]
            mel_filenames, speaker_ids = synth.synthesize(texts, basenames, eval_dir, log_dir, None)

            for elems in zip(texts, mel_filenames, speaker_ids):
                file.write('|'.join([str(x) for x in elems]) + '\n')
    log('synthesized mel spectrograms at {}'.format(eval_dir))
    return eval_dir

def run_eval_folder(args, checkpoint_path, output_dir, hparams, sentences):
    eval_dir = os.path.join(output_dir, 'eval')
    log_dir = os.path.join(output_dir, 'logs-eval')



    if args.model == 'Tacotron-2':
        assert os.path.normpath(eval_dir) == os.path.normpath(args.mels_dir)

    #Create output path if it doesn't exist
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'wavs'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'plots-linear'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'plots-mel'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'plots-align'), exist_ok=True)

    log(hparams_debug_string())


    ckpt_list = os.listdir(checkpoint_path)
    ckpt_list = [ckpt[:-5] for ckpt in ckpt_list if ckpt.endswith('.meta')]
    print(f'total {len(ckpt_list)} ckpts to synthesize in {checkpoint_path}')

    wav_dir = os.path.join(log_dir, 'wavs')
    wav_count_dict = {}
    if os.path.exists(wav_dir):
        wav_list = os.listdir(wav_dir)
        wav_list = [wav.split('batch')[0] for wav in wav_list if wav.endswith('.wav')]

        wav_set = set(wav_list)
        wav_count_dict_list = [{wav: wav_list.count(wav)} for wav in wav_set]
        [wav_count_dict.update(_) for _ in wav_count_dict_list]


    #Set inputs batch wise
    sentences = [sentences[i: i+hparams.tacotron_synthesis_batch_size] for i in range(0, len(sentences), hparams.tacotron_synthesis_batch_size)]

    synth = Synthesizer_Split(hparams)

    ckpt_list = sorted(ckpt_list, key= lambda x: int(x.split('-')[-1]))
    ckpt_list = ckpt_list[::-1]

    for i, ckpt in enumerate(ckpt_list):
        m = re.match('tacotron_model.ckpt-(\d+)', ckpt)
        ckpt_step = m.group(1)

        if f'wav-step_{ckpt_step}_' in wav_count_dict.keys():
            if wav_count_dict[f'wav-step_{ckpt_step}_'] == len(sentences) * 2:
                do_synthesize = False
            else:
                do_synthesize = True
        else:
            do_synthesize = True

        if do_synthesize:
            synth.load(os.path.join(checkpoint_path, ckpt))

            log(f'[{i+1}/{len(ckpt_list)}] Starting Synthesis ')
            with open(os.path.join(eval_dir, 'map.txt'), 'w') as file:
                for i, texts in enumerate(tqdm(sentences)):
                    start = time.time()
                    basenames = ['step_{}_batch_{}_sentence_{}'.format(ckpt_step, i, j) for j in range(len(texts))]
                    mel_filenames, speaker_ids = synth.synthesize(texts, basenames, eval_dir, log_dir, None)

                    for elems in zip(texts, mel_filenames, speaker_ids):
                        file.write('|'.join([str(x) for x in elems]) + '\n')

        else:
            log(f'[{i+1}/{len(ckpt_list)}] ckpt {ckpt} already synthesized in output_{args.name} folder, continue...')
            continue




    return eval_dir

def run_synthesis(args, checkpoint_path, output_dir, hparams):
    GTA = (args.GTA == 'True')
    if GTA:
        synth_dir = os.path.join(output_dir, 'gta')

        #Create output path if it doesn't exist
        os.makedirs(synth_dir, exist_ok=True)
    else:
        synth_dir = os.path.join(output_dir, 'natural')

        #Create output path if it doesn't exist
        os.makedirs(synth_dir, exist_ok=True)


    metadata_filename = os.path.join(args.input_dir, 'train.txt')
    log(hparams_debug_string())
    synth = Synthesizer()
    synth.load(checkpoint_path, hparams, gta=GTA)
    with open(metadata_filename, encoding='utf-8') as f:
        metadata = [line.strip().split('|') for line in f]
        frame_shift_ms = hparams.hop_size / hparams.sample_rate
        hours = sum([int(x[4]) for x in metadata]) * frame_shift_ms / (3600)
        log('Loaded metadata for {} examples ({:.2f} hours)'.format(len(metadata), hours))

    #Set inputs batch wise
    metadata = [metadata[i: i+hparams.tacotron_synthesis_batch_size] for i in range(0, len(metadata), hparams.tacotron_synthesis_batch_size)]

    log('Starting Synthesis')
    mel_dir = os.path.join(args.input_dir, 'mels')
    wav_dir = os.path.join(args.input_dir, 'audio')
    with open(os.path.join(synth_dir, 'map.txt'), 'w') as file:
        for i, meta in enumerate(tqdm(metadata)):
            texts = [m[5] for m in meta]
            mel_filenames = [os.path.join(mel_dir, m[1]) for m in meta]
            wav_filenames = [os.path.join(wav_dir, m[0]) for m in meta]
            basenames = [os.path.basename(m).replace('.npy', '').replace('mel-', '') for m in mel_filenames]
            mel_output_filenames, speaker_ids = synth.synthesize(texts, basenames, synth_dir, None, mel_filenames)

            for elems in zip(wav_filenames, mel_filenames, mel_output_filenames, speaker_ids, texts):
                file.write('|'.join([str(x) for x in elems]) + '\n')
    log('synthesized mel spectrograms at {}'.format(synth_dir))
    return os.path.join(synth_dir, 'map.txt')

def tacotron_synthesize(args, hparams, checkpoint, sentences=None):
    output_dir = 'tacotron_' + args.output_dir
    if args.mode == 'eval_folder':
        output_dir = os.path.join(output_dir, 'output_'+ os.path.basename(os.path.abspath(args.checkpoint)))
        print(f'save output in output_{os.path.basename(os.path.abspath(args.checkpoint))}')
    else:
        output_dir = os.path.join(output_dir, 'output_'+args.name)
    try:
        # checkpoint_path = tf.train.get_checkpoint_state(checkpoint).model_checkpoint_path
        checkpoint_path = args.checkpoint
        log('loaded model at {}'.format(checkpoint_path))
    except:
        raise RuntimeError('Failed to load checkpoint at {}'.format(checkpoint))

    if hparams.tacotron_synthesis_batch_size < hparams.tacotron_num_gpus:
        raise ValueError('Defined synthesis batch size {} is smaller than minimum required {} (num_gpus)! Please verify your synthesis batch size choice.'.format(
            hparams.tacotron_synthesis_batch_size, hparams.tacotron_num_gpus))

    if hparams.tacotron_synthesis_batch_size % hparams.tacotron_num_gpus != 0:
        raise ValueError('Defined synthesis batch size {} is not a multiple of {} (num_gpus)! Please verify your synthesis batch size choice!'.format(
            hparams.tacotron_synthesis_batch_size, hparams.tacotron_num_gpus))

    if args.mode == 'eval':
        return run_eval(args, checkpoint_path, output_dir, hparams, sentences)
    elif args.mode == 'eval_folder':
        return run_eval_folder(args, checkpoint_path, output_dir, hparams, sentences)
    elif args.mode == 'eval_experiment':
        return run_eval_experiment(args, checkpoint_path, output_dir, hparams, sentences)
    elif args.mode == 'synthesis':
        return run_synthesis(args, checkpoint_path, output_dir, hparams)
    else:
        run_live(args, checkpoint_path, hparams)
