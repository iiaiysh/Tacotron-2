import os
from tqdm import tqdm

total_path = '/raid1/stephen/rayhane-tc2/Tacotron-2/training_data/trainingdata_cir/train.txt'
true_false_path = '/raid1/stephen/cirysh_1121.csv'

new_path = '/raid1/stephen/rayhane-tc2/Tacotron-2/training_data/trainingdata_cir/cirysh1104.txt'


f_total = open(total_path, 'r')
f_true_false = open(true_false_path, 'r')


lines_total = f_total.readlines()
lines_true_false = f_true_false.readlines()

fw = open(new_path, 'w')

for i, line in enumerate(tqdm(lines_true_false)):
    if line.split('|')[0] != 'True':
        continue
    text = line.split('|')[4]

    searched = False
    for search_line in lines_total:
        if text in search_line:
            fw.write(search_line)

            searched = True
            break

    if not searched:
        print('not found ',text)