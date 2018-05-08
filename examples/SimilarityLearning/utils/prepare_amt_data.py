# Enlist images in the annotated AMT database along with spatial relation labels
# Author: Akanksha Saran <asaran@cs.utexas.edu>
import os
import json
import numpy as np
from argparse import ArgumentParser
from os import listdir
from os.path import isfile, isdir, join, exists
import csv
from ast import literal_eval
from random import shuffle
from PIL import Image  
from math import floor

def main(hparams):
    hits_dir = hparams.input_hits_dir
    save_dir = hparams.save_dir
    feats_path = hparams.feats_path
    img_dir = hparams.img_dir

    relation_dict = {                
        'on':0,
        'in':1,
        'near':2,
        'beside':3,
        'next to':4,
        'to the left of':5,
        'to the right of':6,
        'below':7,
        'above':8,
        'at':9,
        'behind':10,
        'on top of':11
    }

    outfiles = ['../data/amt_train.json','../data/amt_test.json']
    # Open all CSV files to get total images - split into 80-20 as train, val
    # Open CSV file to read image name, bounding box coordinates, relation name.
    # Read image and store height and width of image
    # Normalize bounding box coordinates
    # Save every entry in a json

    #total_imgs = 0
    imgs = []
 
    batch_csvs = [batch for batch in listdir(hits_dir) if (isfile(join(hits_dir,batch)) and batch.endswith('.csv'))]
    for batch in batch_csvs:
        infile = open(join(hits_dir,batch), 'rb')
        reader = csv.reader(infile, delimiter=",")
        header_row = next(reader) 
        img_idx = header_row.index('Input.image_url')
        #approve_idx = header_row.index('Approve')
        bb_idx = header_row.index('Answer.annotation_data')
        for row in reader:
            bbs = literal_eval(row[bb_idx])
            empty_dict = not(bool(bbs))
            if(not empty_dict):# and row[approve_idx]=='x'):
                img_url = row[img_idx]
                im = img_url.split('/')
                img_name = im[-1]
                #total_imgs = total_imgs+ 1
                if(img_name not in imgs):
                    imgs.append(img_name)

    total_imgs = len(imgs)   
    training_img_num = int(floor(0.8*total_imgs))
    validation_img_num = total_imgs - training_img_num
    shuffle(imgs)

    training_imgs = imgs[0:training_img_num]
    validation_imgs = imgs[training_img_num+1:]

    tr_data = []
    val_data = []
    for batch in batch_csvs:
        infile = open(join(hits_dir,batch), 'rb')
        reader = csv.reader(infile, delimiter=",")
        header_row = next(reader) 
        img_idx = header_row.index('Input.image_url')
        #approve_idx = header_row.index('Approve')
        bb_idx = header_row.index('Answer.annotation_data')
        spatial_relation_idx = header_row.index('Input.objects_to_find')
        for row in reader:
            bbs = literal_eval(row[bb_idx])
            empty_dict = not(bool(bbs))
            if(not empty_dict):# and row[approve_idx]=='x'):
                img_url = row[img_idx]
                im = img_url.split('/')
                img_name = im[-1]
                image_file = join(img_dir,img_name)
                img_content = Image.open(image_file)
                im_size = img_content.size
                width = im_size[0]
                height = im_size[1]
                if len(bbs)==2:
                    obj_bb = bbs[0]
                    subj_bb = bbs[1]

                    obj_bb_list = [ (obj_bb['left']/float(width)),  (obj_bb['top']/float(height)),  ((obj_bb['left']+obj_bb['width'])/float(width)),  (obj_bb['top']/float(height)), \
                                    ((obj_bb['left']+obj_bb['width'])/float(width)),  ((obj_bb['top']+obj_bb['height'])/float(height)),  (obj_bb['left']/float(width)),  ((obj_bb['top']+obj_bb['height'])/float(height))]
                    subj_bb_list = [ (subj_bb['left']/float(width)),  (subj_bb['top']/float(height)),  ((subj_bb['left']+subj_bb['width'])/float(width)),  (subj_bb['top']/float(height)), \
                                     ((subj_bb['left']+subj_bb['width'])/float(width)),  ((subj_bb['top']+subj_bb['height'])/float(height)),  (subj_bb['left']/float(width)),  ((subj_bb['top']+subj_bb['height'])/float(height))]

                    bb_list = obj_bb_list + subj_bb_list
                    #print(type(bb_list[0]))
                    label = row[spatial_relation_idx]

                    data_dict = {}
                    data_dict['img_name'] = img_name
                    data_dict['bb'] = bb_list
                    data_dict['label'] = relation_dict[label]
                    data_dict['feat_path'] = feats_path

                    if(img_name in training_imgs):
                    	tr_data.append(data_dict)
                    elif(img_name) in validation_imgs:
                    	val_data.append(data_dict)
    
    with open(outfiles[0], 'w') as outfile:
       	json.dump(tr_data, outfile)
    with open(outfiles[1], 'w') as outfile:
       	json.dump(val_data, outfile)

    #print len(imgs)
    print training_img_num
    print validation_img_num
    print len(tr_data)
    print len(val_data)

if __name__ == '__main__':
    PARSER = ArgumentParser()
    PARSER.add_argument('--feats-path', type=str, default='/home/siml/Documents/spatial-relations/tensorpack_old/examples/SimilarityLearning/features/VGGNet/fc7/amt_data.pkl',help='directory containing pretrained CNN features')
    PARSER.add_argument('--input-hits-dir', type=str, default='/home/siml/Documents/spatial-relations/AMT-spatial-relations/HITS/',help='directory containing approved CSVs')
    PARSER.add_argument('--img-dir', type=str, default='/home/siml/Documents/spatial-relations/tensorpack_old/examples/SimilarityLearning/data/AMT/imgs/',help='directory containing approved CSVs')
    PARSER.add_argument('--save-dir', type=str, default='/home/siml/Documents/spatial-relations/tensorpack_old/examples/SimilarityLearning/data/AMT/', help='directory where input data for the network should be written')

    HPARAMS = PARSER.parse_args()
    main(HPARAMS)