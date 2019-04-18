import os
import pygame
pygame.mixer.pre_init(44100,-16,2,2048)
pygame.init()
pygame.mixer.init()

src_file = '/raid1/stephen/data/new_annotation_0307/default-missed_label-tony_pure_book-special_charactor-none_eng_punc-duration-cmudict.csv'
split_mark = '|'
name_part = 0
text_part = 3
name_prefix = '/raid1/stephen/data/new_annotation_0307/wavs-default'
name_postfix = '.wav'

lines = open(src_file, 'r').readlines()

lines_len = len(lines)

print(f'total lines {lines_len}')

done_lines = []
if os.path.exists(f'{src_file}.record.csv'):
    done_lines = open(f'{src_file}.record.csv', 'r').readlines()

fw = open(f'{src_file}.record.csv', 'a')

for i, line in enumerate(lines):
    if line in done_lines:
        continue

    parts = line.strip().split(split_mark)

    audio_path = os.path.join(name_prefix, f'{parts[name_part]}{name_postfix}')

    if os.path.isfile(audio_path):
        pygame.mixer.music.load(audio_path)

    text = parts[text_part]

    print(f'[{i}]/[{lines_len}] {text}')

    while True:
        pygame.mixer.music.play(1)
        k = input('Is this sample match the text? [y:yes]/[n:no]')
        if k == 'y':
            match = True
            break
        elif k == 'n':
            match = False
            break
        else:
            pass
    print('\n')

    fw.write(f'{match}|{line}')