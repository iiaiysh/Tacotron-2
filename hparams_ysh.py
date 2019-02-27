import numpy as np
import tensorflow as tf

_tacotron_num_gpus = 2
# Default hyperparameters
hparams_ysh = tf.contrib.training.HParams(
	max_mel_frames = 1500,  #Only relevant when clip_mels_length = True, please only use after trying output_per_steps=3 and still getting OOM errors.
	tacotron_num_gpus=_tacotron_num_gpus,  # Determines the number of gpus in use for Tacotron training.
	tacotron_batch_size=32*_tacotron_num_gpus,  # number of training samples on each training steps, should be N*32
	tacotron_reg_weight=1e-6,  # regularization weight (for L2 regularization)
	tacotron_synthesis_batch_size=1, # DO NOT MAKE THIS BIGGER THAN 1 IF YOU DIDN'T TRAIN TACOTRON WITH "mask_encoder=True"!!
	tacotron_dropout_rate_synthesis = 0.5,
	tacotron_fine_tuning = False, #Set to True to freeze encoder and only keep training pretrained decoder. Used for speaker adaptation with small data.
	tacotron_initial_learning_rate=3e-4,  # starting learning rate, 3e-4 pick for the blizzard 90k finetune
	tacotron_final_learning_rate = 1e-4, #minimal learning rate

	tacotron_start_decay=40000 + 90000,  # Step at which learning decay starts
	find_lr = False, #whether to use cyclic lr find
	test_lr = False,
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