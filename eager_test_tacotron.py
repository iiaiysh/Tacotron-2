import pickle
from tacotron.models import create_model_ysh
from hparams import hparams
from hparams_ysh import hparams_ysh, update_hp1_with_hp2
import tensorflow as tf

tf.enable_eager_execution()

tf.executing_eagerly()

f = open('feedr.file','rb')

feeder = pickle.load(f)

update_hp1_with_hp2(hparams, hparams_ysh)


global_step = tf.Variable(0, name='global_step', trainable=False)

model = create_model_ysh('Tacotron', hparams)

model.initialize(feeder['inputs'], feeder['input_lengths'], feeder['mel_targets'], feeder['token_targets'], linear_targets=feeder['linear_targets'],
				targets_lengths=feeder['targets_lengths'], global_step=global_step,
				is_training=True, split_infos=feeder['split_infos'])