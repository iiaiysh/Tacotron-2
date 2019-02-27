from .tacotron import Tacotron
from .tacotron_ysh import Tacotron_ysh

def create_model(name, hparams):
  if name == 'Tacotron':
    return Tacotron(hparams)
  else:
    raise Exception('Unknown model: ' + name)


def create_model_ysh(name, hparams):
  if name == 'Tacotron':
    return Tacotron_ysh(hparams)
  else:
    raise Exception('Unknown model: ' + name)