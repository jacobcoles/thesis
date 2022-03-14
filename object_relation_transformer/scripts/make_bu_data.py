from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import argparse


parser = argparse.ArgumentParser()

# output_dir
parser.add_argument('--downloaded_feats', default='./data/sentences3Dfrcnn', help='downloaded feature directory')
parser.add_argument('--output_dir', default='data/cocobu', help='output feature files')

args = parser.parse_args()
params = vars(args)

csv.field_size_limit(sys.maxsize)


FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
"""
infiles = ['trainval/karpathy_test_resnet101_faster_rcnn_genome.tsv'],
          'trainval/karpathy_val_resnet101_faster_rcnn_genome.tsv',\
          'trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.0', \
          'trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.1']
"""

path_boi = params['downloaded_feats']

infiles = [file for file in os.listdir(path_boi) if '.npz' in str(file)]


"""
for infile in infiles:
    print('Reading ' + infile)
    with open(os.path.join(args.downloaded_feats, infile), "r+") as tsv_in_file:
"""
os.makedirs(args.output_dir+'_att')
os.makedirs(args.output_dir+'_fc')
os.makedirs(args.output_dir+'_box')

if not os.path.isdir(args.output_dir+'_att'):
    os.mkdir(args.output_dir+'_att')
if not os.path.isdir(args.output_dir+'_fc'):
    os.mkdir(args.output_dir+'_fc')
if not os.path.isdir(args.output_dir+'_box'):
    os.mkdir(args.output_dir+'_box')

for infile in infiles:
    item = np.load(os.path.join(params['downloaded_feats'], infile))
    image_id = int(item['img_id'])
    num_boxes = int(item['num_boxes'])
    boxes = np.frombuffer(base64.b64decode(np.frombuffer(item['boxes'])), 
                        dtype=np.float32).reshape((item['num_boxes'],-1))
    features = np.frombuffer(base64.b64decode(np.frombuffer(item['features'])), 
                        dtype=np.float32).reshape((item['num_boxes'],-1))

    np.savez_compressed(os.path.join(args.output_dir+'_att', str(image_id)), feat=features)
    np.save(os.path.join(args.output_dir+'_fc', str(image_id)), features.mean(0))
    np.save(os.path.join(args.output_dir+'_box', str(image_id)), boxes)



