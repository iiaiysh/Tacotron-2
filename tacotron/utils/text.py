import re

from . import cleaners
from .symbols import symbols

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

def line_split_3(text):
    assert type(text) == str
    if len(text.split(' ')) <= 20:
        return [text]
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

def line_split_2(text):
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


# three principle:
# 1.split sentence with .?!
# 2.every sentence not too loog (<20)
# 2.every sentence not too short (>5)

def line_split(text):
    class Exception_No_Last_Item(Exception):pass

    class Exception_Empty_List(Exception): pass

    assert type(text) == str
    assert text.strip() != ''

    # principle 1
    split_marks = '.!?'
    text_list = re.findall(f'([^{split_marks}]+|[{split_marks}])', text)
    if text_list[0] in split_marks:
      text_list = text_list[1:]

    if len(text_list) % 2 ==0:
      text_split_by_dot = [text_list[i]+text_list[i+1] for i in range(0, len(text_list), 2)]
    else:
      text_split_by_dot = [text_list[i]+text_list[i+1] for i in range(0, len(text_list)-1, 2)]
      text_split_by_dot.append(text_list[-1]+'.')


    # principle 2
    text_split_final = []
    for part in text_split_by_dot:
      # less then 20 words
      if len(part.split(' ')) < 20:
        text_split_final.append(part)
        continue
      else:
        part_split_by_comma = part.split(',')

        text_concate = []
        for item in part_split_by_comma:
          text_concate.append(item)
          if len(''.join(text_concate).split(' '))>20:
            if len(text_concate) == 1:
              text_split_final.append(text_concate[0]+',')
              text_concate = []
            else:
              text_split_final.append(', '.join(text_concate[:-1]) + ',')
              text_concate = [item]
        if len(text_concate) != 0:
          text_split_final.append(', '.join(text_concate))

    #principle 3
    text_split_final_final = []
    last_item = ''
    for item in text_split_final:
        if len((last_item +' '+item).strip().split(' ')) < 5:
            last_item = last_item +' '+item
            continue
        else:
            text_split_final_final.append(last_item+' '+item)
            last_item = ''
    try:
        if last_item == '':
            raise Exception_No_Last_Item()

        if text_split_final_final == []:
            raise Exception_Empty_List()

        item_tmp = text_split_final_final[-1] + ' ' + last_item
        text_split_final_final.pop()
        text_split_final_final.append(item_tmp)

    except Exception_No_Last_Item:
        pass
    except Exception_Empty_List:
        text_split_final_final.append(last_item)

    assert type(text_split_final_final) == list
    assert len(text_split_final_final) > 0
    return text_split_final_final


def text_to_sequence(text, cleaner_names):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through

    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []

  # Check for curly braces and treat their contents as ARPAbet:
  while len(text):
    m = _curly_re.match(text)
    if not m:
      sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
      break
    sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
    sequence += _arpabet_to_sequence(m.group(2))
    text = m.group(3)

  # Append EOS token
  sequence.append(_symbol_to_id['~'])
  return sequence


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      # Enclose ARPAbet back in curly braces:
      if len(s) > 1 and s[0] == '@':
        s = '{%s}' % s[1:]
      result += s
  return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text


def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
  return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
  return s in _symbol_to_id and s is not '_' and s is not '~'
