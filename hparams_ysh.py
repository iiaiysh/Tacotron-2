import numpy as np
import tensorflow as tf

# Default hyperparameters
hparams_ysh = tf.contrib.training.HParams(
	max_mel_frames = 1700,  #Only relevant when clip_mels_length = True, please only use after trying output_per_steps=3 and still getting OOM errors.
	tacotron_num_gpus=2,  # Determines the number of gpus in use for Tacotron training.
	tacotron_batch_size=32*2,  # number of training samples on each training steps, should be N*32
	tacotron_reg_weight=1e-6,  # regularization weight (for L2 regularization)

)

def hparams_debug_string():
	values = hparams_ysh.values()
	hp = ['  %s: %s' % (name, values[name]) for name in sorted(values) if name != 'sentences']
	return 'Hyperparameters:\n' + '\n'.join(hp)
