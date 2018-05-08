import numpy as np
from argparse import ArgumentParser
import os
from os import listdir
from os.path import isfile, isdir, join, exists
import csv
from ast import literal_eval

# read all batch csvs in HITS directory
# accept images which are approved in csvs
# create a txt with img file names

def main(hparams):
    hits_dir = hparams.input_hits_dir
    save_file = hparams.save_file

    imgs = []
    f=open(save_file,'w')
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
            if(not empty_dict): # and row[approve_idx]=='x'):
                img_url = row[img_idx]
                im = img_url.split('/')
                img_name = im[-1]
                if(img_name not in imgs):
                    imgs.append(img_name)
                    f.write('/home/siml/Documents/spatial-relations/tensorpack_old/examples/SimilarityLearning/data/AMT/imgs/'+ img_name + "\n")
    f.close()

if __name__ == '__main__':
    PARSER = ArgumentParser()
    PARSER.add_argument('--input-hits-dir', type=str, default='/home/siml/Documents/spatial-relations/AMT-spatial-relations/HITS/',help='directory containing approved CSVs')
    PARSER.add_argument('--save-file', type=str, default='../../../data/AMT/amt_data.txt', help='directory where input data for the network should be written')

    HPARAMS = PARSER.parse_args()
    main(HPARAMS)