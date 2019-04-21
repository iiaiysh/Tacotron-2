'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''

import re

from unidecode import unidecode

from .numbers import normalize_numbers

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]

# List of (regular expression, replacement) pairs for abbreviations:
_yshmarks_1 = [(re.compile('\\b%s\\b' % x[0], re.IGNORECASE), x[1]) for x in [
  ('IBM', 'II BE eMm'),
  ('OK', 'okey'),
  ('idea', 'idearr'),
  ('cool', 'cruel'),
  ('truth', 'truthh'),
  ('hi', 'hey'),
  ('error', 'eyeroow'),
  ('Neha', 'Nee haa'),
  ('childhood', 'chaild huodd'),
  ('today', 'todayy'),

]]

_yshmarks_2 = [(re.compile('\\b%s\\b' % x[0],), x[1]) for x in [
  ('AM', 'ay em'),
  ('PM', 'pee eMm'),
  ('MSDOS', 'eMm eS DaoS'),

]]

def expand_yshmarks(text):
  for regex, replacement in _yshmarks_1:
    text = re.sub(regex, replacement, text)
  for regex, replacement in _yshmarks_2:
    text = re.sub(regex, replacement, text)
  return text

def expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text


def expand_numbers(text):
  return normalize_numbers(text)


def lowercase(text):
  '''lowercase input tokens.
  '''
  return text.lower()


def collapse_whitespace(text):
  return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
  return unidecode(text)


def basic_cleaners(text):
  '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def transliteration_cleaners(text):
  '''Pipeline for non-English text that transliterates to ASCII.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def english_cleaners(text):
  '''Pipeline for English text, including number and abbreviation expansion.'''
  text = convert_to_ascii(text)
  # text = lowercase(text)
  text = expand_numbers(text)
  text = expand_abbreviations(text)
  text = collapse_whitespace(text)
  return text

def ysh_cleaners(text):
  '''Pipeline for English text, including number and abbreviation expansion.'''
  text = convert_to_ascii(text)
  # text = lowercase(text)
  text = expand_numbers(text)
  text = expand_abbreviations(text)
  text = collapse_whitespace(text)

  text = expand_yshmarks(text)

  # add stop mark in the last if no punctuation
  if re.match('[a-zA-Z]', text[-1]):
    text = text + '.'
  print(f'cleaned: {text}')
  return text
