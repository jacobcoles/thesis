import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import json
import glob

json_boi = dict()
json_boi['dataset'] = "SentencesNYUv2"
json_boi['images'] = list()

imgs_len = len(glob.glob1('../SentencesNYUv2_toolbox/data/descriptions_info/',"*.mat"))
train_percent = 0.7
val_percent = 0.15

train_ = ['train' for i in range(int(imgs_len*train_percent))]
val_ = ['val' for i in range(int(imgs_len*val_percent))]
test_ = ['test' for i in range(imgs_len-len(train_)-len(val_))]
tvt_split = train_ + val_ + test_

sentid = 0
for i, filename in enumerate(glob.glob1('../SentencesNYUv2_toolbox/data/descriptions_info/',"*.mat")):
    imgid = int(filename.strip('.mat').strip('in'))

    append_item = dict()
    append_item['sentids'] = list()
    append_item['imgid'] = imgid
    append_item['sentences'] = list()
    append_item['split'] = tvt_split[i]
    append_item['filepath'] = '../SentencesNYUv2_toolbox/data/images'
    append_item['filename'] = str(imgid).zfill(4) + '.jpg'
    append_item['cocoid'] = imgid

    image_descriptions = loadmat(f'../SentencesNYUv2_toolbox/data/descriptions_info/in{str(imgid).zfill(4)}.mat')['descriptions'][0]
    sentences_list = [thing.strip(' ')+' .' for thing in image_descriptions[0][0][0].split('.')][:-1]

    """ #makes it work with 5 sentences to match coco format
    for sent_num in range(5):
        sent_idx = sent_num % len(sentences_list)

        sentence_dict = dict()
        sentence = sentences_list[sent_idx]

        sentence_dict['tokens'] = sentence.strip('.').split(' ')
        sentence_dict['raw'] = sentence
        sentence_dict['imageid'] = imgid
        sentence_dict['sentid'] = sentid

        append_item['sentences'].append(sentence_dict)

        append_item['sentids'].append(sentid)
        sentid += 1
    """
                                        
    sentence_dict = dict()
    sentence = ' '.join(sentences_list)

    sentence_dict['tokens'] = sentence.split(' ')
    sentence_dict['raw'] = sentence
    sentence_dict['imageid'] = imgid
    sentence_dict['sentid'] = sentid

    append_item['sentences'].append(sentence_dict)

    append_item['sentids'].append(sentid)
    sentid += 1

    json_boi['images'].append(append_item)
            
with open('dataset_annotations_.json', 'w') as f:
    json.dump(json_boi, f)