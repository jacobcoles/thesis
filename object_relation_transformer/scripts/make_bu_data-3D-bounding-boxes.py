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
import h5py
from PIL import Image
from tqdm import tqdm

#almost the same as the code a bit further up
def getObjectMasks(imgObjectLabels, imgInstances):
    #size of the image
    H, W = imgObjectLabels.shape

    #Shaping our dimensions, so that we can put the instance and object labels 'beside' each other
    imgObjectLabelsExpDim = np.expand_dims(imgObjectLabels, axis=2)
    imgInstancesExpDim = np.expand_dims(imgInstances, axis=2)

    #Concatenate the object labels with the instance labels. 
    #Each 'pixel' will have both a value for the object and instance. (shape of (W, H, 2))
    imgObjectAndInstances = np.concatenate([imgObjectLabelsExpDim, imgInstancesExpDim], axis=2)
    imgObjectAndInstances = imgObjectAndInstances.reshape(-1, 2)

    #unique combinations of object and instance to get all the individual object (labels)
    pairs = np.unique(imgObjectAndInstances, axis = 0)
    N = pairs.shape[0]

    #prepare the masks where we will get the mask for each instance of object
    instanceMasks = np.zeros((H, W, N)).astype(bool)
    instanceLabels = np.zeros((N, 1)).astype(bool)
    for i in range(N):
        imgObjectLabelsTruth = np.where(imgObjectLabels == pairs[i,0], 1, 0)
        imgInstancesTruth = np.where(imgInstances == pairs[i,1], 1, 0)

        instanceMasks[:,:,i] = np.where(imgObjectLabelsTruth & imgInstancesTruth, 1, 0)
        instanceLabels[i] = pairs[i,0]
        
    instanceMasksReshape = np.transpose(instanceMasks,(2,1,0))
    
    return instanceMasksReshape

def create3dBoxFromRcnnWithNyuAsDepth(nyu_dataset_dict, image_id, frcnn_boxes):
    data_dict = nyu_dataset_dict

    #do this for all images. Note that image_id and image_index are one off, because reasons. 
    image_index = image_id-1

    #an_image = np.transpose(data_dict['images'][image_index], (2,1,0))
    an_image_depth = np.transpose(data_dict['depths'][image_index], (1,0))

    img_object_labels = data_dict['labels'][image_index]
    img_instances = data_dict['instances'][image_index]

    #We get a boolean "image" mask, showing where each object in the ground-truth is
    ground_truth_masks = getObjectMasks(img_object_labels, img_instances)

    H, W = an_image_depth.shape
    total_pixels = H*W

    new_3D_frcnn_boxes = np.zeros((frcnn_boxes.shape[0], 6))
    new_3D_frcnn_boxes[:,0:2] = frcnn_boxes[:,0:2]
    new_3D_frcnn_boxes[:,3:5] = frcnn_boxes[:,2:4]

    #We then go through the frcnn boxes and make an equivalent image mask based on these
    for frcnn_index, box in enumerate(frcnn_boxes):
        best_match_ground_truth_index = int()
        best_overlap_proportion = float()

        frcnn_mask = np.zeros((H, W)).astype(bool)
        frcnn_mask[box[1]:box[3],box[0]:box[2]] = 1

        #We see where the biggest (proportional) overlap for each F-R-CNN box is with each of the ground-truth boxes
        #This is to work out which f-r-cnn box corresponds with which mask in the ground truth to correctly 
        #    get the 3D bounding box for the each object

        for ground_truth_index, ground_truth_mask in enumerate(ground_truth_masks):
            proportion = np.sum(np.dot(frcnn_mask.flatten(), ground_truth_mask.flatten()))/total_pixels
            if proportion > best_overlap_proportion:
                best_match_ground_truth_index = ground_truth_index
                best_overlap_proportion = proportion

        #Now we need to get the maximum and minimum 3D values for the object from the depth map/image in the ground truth
        image_coordinates_for_mask = np.nonzero(ground_truth_masks[best_match_ground_truth_index])
        depth_values_for_this_img_mask = an_image_depth[image_coordinates_for_mask]
        minimum_depth = depth_values_for_this_img_mask.min()
        maximum_depth = depth_values_for_this_img_mask.max()

        #And we inject that alongside the frcnn data_
        new_3D_frcnn_boxes[frcnn_index, 2] = minimum_depth
        new_3D_frcnn_boxes[frcnn_index, 5] = maximum_depth

    return new_3D_frcnn_boxes

parser = argparse.ArgumentParser()

# output_dir
parser.add_argument('--downloaded_feats', default='./data/sentences3Dfrcnn', help='downloaded feature directory')
parser.add_argument('--output_dir', default='data/cocobu-3D-boxes', help='output feature files')

args = parser.parse_args()
params = vars(args)

csv.field_size_limit(sys.maxsize)


FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

path_boi = params['downloaded_feats']

infiles = [file for file in os.listdir(path_boi) if '.npz' in str(file)]

if not os.path.isdir(args.output_dir+'_att'):
    os.mkdir(args.output_dir+'_att')
if not os.path.isdir(args.output_dir+'_fc'):
    os.mkdir(args.output_dir+'_fc')
if not os.path.isdir(args.output_dir+'_box'):
    os.mkdir(args.output_dir+'_box')
    
nyu_dataset_path = '../nyu_depth_v2_labeled.mat'
nyu_dataset_dict = h5py.File(nyu_dataset_path, 'r')

for infile in tqdm(infiles):
    item = np.load(os.path.join(params['downloaded_feats'], infile))
    image_id = int(item['img_id'])
    num_boxes = int(item['num_boxes'])
    boxes = np.frombuffer(base64.b64decode(np.frombuffer(item['boxes'])), 
                        dtype=np.float32).reshape((item['num_boxes'],-1))
    boxes = boxes.astype(int)
    
    boxes_3d = create3dBoxFromRcnnWithNyuAsDepth(nyu_dataset_dict, image_id, boxes)
    
    features = np.frombuffer(base64.b64decode(np.frombuffer(item['features'])), 
                        dtype=np.float32).reshape((item['num_boxes'],-1))

    np.savez_compressed(os.path.join(args.output_dir+'_att', str(image_id)), feat=features)
    np.save(os.path.join(args.output_dir+'_fc', str(image_id)), features.mean(0))
    np.save(os.path.join(args.output_dir+'_box', str(image_id)), boxes_3d)


    

