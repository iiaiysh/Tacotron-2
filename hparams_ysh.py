import numpy as np
import tensorflow as tf

_tacotron_num_gpus = 1
_wavenet_num_gpus = 1
# Default hyperparameters
hparams_ysh = tf.contrib.training.HParams(

	is_training=True,

	tacotron_fine_tuning=False, #Set to True to freeze encoder and only keep training pretrained decoder. Used for speaker adaptation with small data.

	synthesis_constraint=False,
	# Whether to use attention windows constraints in synthesis only (Useful for long utterances synthesis)

	tacotron_test_size=0.05,
	# % of data to keep as test data, if None, tacotron_test_batches must be not None. (5% is enough to have a good idea about overfit)
	tacotron_test_batches=None,  # number of test batches.
	tacotron_test_batch_size=32,  # number of training samples on each training steps, should be N*32


	fmin=55,	# Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
	fmax=7600,  # To be increased/reduced depending on data.
	rescale = False, #Whether to rescale audio prior to preprocessing
	preemphasis = 0.97, #filter coefficient.
	magnitude_power = 1., #The power of the spectrogram magnitude (1. for energy, 2. for power)



	max_mel_frames = 1500,  #Only relevant when clip_mels_length = True, please only use after trying output_per_steps=3 and still getting OOM errors.
	tacotron_num_gpus=_tacotron_num_gpus,  # Determines the number of gpus in use for Tacotron training.
	tacotron_batch_size=32*_tacotron_num_gpus,  # number of training samples on each training steps, should be N*32
	# tacotron_reg_weight=1e-5,  # regularization weight (for L2 regularization)
	tacotron_synthesis_batch_size=1, # DO NOT MAKE THIS BIGGER THAN 1 IF YOU DIDN'T TRAIN TACOTRON WITH "mask_encoder=True"!!
	tacotron_dropout_rate_synthesis = 0.5,
	tacotron_initial_learning_rate=1e-3,  # starting learning rate, 3e-4 pick for the blizzard 90k finetune
	tacotron_final_learning_rate = 1e-4, #minimal learning rate

	experiment_preprocess=False,#whther to experiment preprocess

	tacotron_start_decay=40000+158000,  # Step at which learning decay starts

	find_lr = False, #whether to use cyclic lr find
	findlr_initial_learning_rate=1e-6,
	findlr_speed=1.02,

	test_lr = False,
	test_lr_min = 1e-5,
	test_lr_max = 1e-3,
	test_lr_stepsize_num_epoch = 5,
	test_lr_one_epoch_step = 20,
	max_iters = 5000,


	wavenet_num_gpus=_wavenet_num_gpus, #Determines the number of gpus in use for WaveNet training.
	wavenet_batch_size=8*_wavenet_num_gpus, #batch size used to train wavenet.

	sentences = [
	# From July 8, 2017 New York Times:
	'Scientists at the CERN laboratory say they have discovered a new particle.',
	'There\'s a way to measure the acute emotional intelligence that has never gone out of style.',
	'President Trump met with other leaders at the Group of 20 conference.',
	'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',
	# From Google's Tacotron example page:
	'Generative adversarial network or variational auto-encoder.',
	'Basilar membrane and otolaryngology are not auto-correlations.',
	'He has read the whole thing.',
	'He reads books.',
	'He thought it was time to present the present.',
	'Thisss isrealy awhsome.',
	'The big brown fox jumps over the lazy dog.',
	'Did the big brown fox jump over the lazy dog?',
	"Peter Piper picked a peck of pickled peppers. How many pickled peppers did Peter Piper pick?",
	"She sells sea-shells on the sea-shore. The shells she sells are sea-shells I'm sure.",
	"Tajima Airport serves Toyooka.",
	#From The web (random long utterance)
	# 'On offering to help the blind man, the man who then stole his car, had not, at that precise moment, had any evil intention, quite the contrary, \
	# what he did was nothing more than obey those feelings of generosity and altruism which, as everyone knows, \
	# are the two best traits of human nature and to be found in much more hardened criminals than this one, a simple car-thief without any hope of advancing in his profession, \
	# exploited by the real owners of this enterprise, for it is they who take advantage of the needs of the poor.',
	# A final Thank you note!
	'Thank you so much for your support!',
	'Welcome to the inception institute of artificial intelligence, this is Tony Robbins, I am kidding, this is not really Tony Robbins, this is a computer generated voice, I am still a work in progress so do not mind any strange artifacts that you hear.',
	#0307 samples
	"The more grateful we are, the moment you feel total grattitude, to life, to God, to your friends, to your family, that's the moment you're rich.",
	"That's why I gave you this Ultimate Edge program, so you can just do the first week of personal power then do Get the Edge and then pop in for a couple days with inner strength, use the films!",
	"If a baby is not stroked, isn't physically loved, isn't given as the doctor's say, tactile stimulation, also known as love, then the baby has failure to thrive syndrome.",
	'I am not saying by this tape program that you should never be disappointed, or frustrated, or angry or anything else cause all of those emotions have their place in their proper proportion.',
	'Welcome to the inception institute of artificial intelligence, this is Tony Robbins,',
	'I am kidding, this is not really Tony Robbins, this is a computer generated voice,',
	'I am still a work in progress so do not mind any strange artifacts that you hear.',
	],
)

def hparams_debug_string():
	values = hparams_ysh.values()
	hp = ['  %s: %s' % (name, values[name]) for name in sorted(values) if name != 'sentences']
	return 'Hyperparameters:\n' + '\n'.join(hp)

def update_hp1_with_hp2(hp1, hp2):
	keys1 = hp1.values().keys()
	keys2 = hp2.values().keys()

	for key in keys2:
		if key in keys1:
			hp1.set_hparam(key, hp2.get(key))
		else:
			hp1.add_hparam(key, hp2.get(key))

	# return hp1