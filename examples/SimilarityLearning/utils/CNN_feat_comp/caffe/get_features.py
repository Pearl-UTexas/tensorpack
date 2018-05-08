import numpy as np
import os
import sys
from argparse import ArgumentParser
import sys
import caffe
import pickle as pkl

#save feat dicts with correct key (img name)

def main(hparams):
    # Make sure that caffe is on the python path:
    caffe_root = '/home/siml/caffe/' 
    sys.path.insert(0, caffe_root + 'python')
    pycaffe_dir = os.path.join(caffe_root, 'python')
    
    caffe.set_mode_gpu() #instead of set_mode_cpu()

    model = hparams.model
    feature_type = hparams.feature_type
    feature_dim = hparams.feature_dim
    input_data = hparams.input_data
    save_dir = hparams.save_dir

    if(model=='ResNet-152'):        
        net = caffe.Net('/home/siml/Documents/spatial-relations/tensorpack_old/examples/SimilarityLearning/models/ResNet-152/deploy.prototxt',
                        '/home/siml/Documents/spatial-relations/tensorpack_old/examples/SimilarityLearning/models/ResNet-152/ResNet-152-model.caffemodel',
                        caffe.TEST)
    elif(model=='AlexNet'):
        net = caffe.Net('/home/siml/Documents/spatial-relations/tensorpack_old/examples/SimilarityLearning/models/AlexNet/deploy.prototxt',
                        '/home/siml/Documents/spatial-relations/tensorpack_old/examples/SimilarityLearning/models/AlexNet/bvlc_alexnet.caffemodel',
                        caffe.TEST)
    elif(model=='GoogleNet'):
        net = caffe.Net('/home/siml/Documents/spatial-relations/tensorpack_old/examples/SimilarityLearning/models/GoogleNet/deploy.prototxt',
                        '/home/siml/Documents/spatial-relations/tensorpack_old/examples/SimilarityLearning/models/GoogleNet/bvlc_googlenet.caffemodel',
                        caffe.TEST)
    elif(model=='VGGNet'):
        net = caffe.Net('/home/siml/Documents/spatial-relations/tensorpack_old/examples/SimilarityLearning/models/VGGNet/deploy.prototxt',
                        '/home/siml/Documents/spatial-relations/tensorpack_old/examples/SimilarityLearning/models/VGGNet/VGG_ILSVRC_19_layers.caffemodel',
                        caffe.TEST)
    
    trials = [f for f in os.listdir(input_data) if os.path.isfile(os.path.join(input_data, f))]
    #trials = os.listdir(input_data)
    for trial in trials:
        print(trial)

        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        meanFile = os.path.join(pycaffe_dir, 'caffe/imagenet/ilsvrc_2012_mean.npy')
        transformer.set_mean('data', np.load(meanFile).mean(1).mean(1)) # mean pixel
        transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

        # set net to batch size of 50
        if(model=='AlexNet'):
           net.blobs['data'].reshape(50,3,227,227) #AlexNet 
        else:
            net.blobs['data'].reshape(16,3,224,224)

        test_image_names = []
        test_images = open(os.path.join(input_data,trial), 'r')

        for line in test_images:        
            test_image_names.append(line.rstrip('\n'))

        # assign test labels
        write_dir = os.path.join(save_dir, model + '/' + feature_type)
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        trial_name = trial.split('.txt', 1)[0]
        write_file = write_dir + '/' + trial_name + '.pkl'

        feats_dict = {}
        for loc in test_image_names:
            #print(loc)
            img = caffe.io.load_image(loc);
            net.blobs['data'].data[...] = transformer.preprocess('data', img);
            net.forward();
            feats = net.blobs[feature_type].data[0]
            feats = feats.reshape(feature_dim)
            #print(len(feat_fc6))
            
            local_loc = loc.split('/')
            local_loc = local_loc[-1]
            print(local_loc)
            feats_dict[local_loc] = feats
            #print write_file
       
        pkl.dump(feats_dict, open(write_file,'wb'))


if __name__ == '__main__':
    PARSER = ArgumentParser()

    PARSER.add_argument('--model', type=str, default='AlexNet', help='type of deep model used')
    PARSER.add_argument('--feature-type', type=str, default='fc7', help='feature layer name for which activations are being extracted')
    PARSER.add_argument('--feature-dim', type=int, default=4096, help='dimension of the feature being extracted')
    PARSER.add_argument('--input-data', type=str, default='/home/siml/Documents/spatial-relations/tensorpack_old/examples/SimilarityLearning/data/AMT/', help='file path used as input to the network')
    PARSER.add_argument('--save-dir', type=str, default='/home/siml/Documents/spatial-relations/tensorpack_old/examples/SimilarityLearning/features/', help='directory where computed features will be saved')

    HPARAMS = PARSER.parse_args()

    main(HPARAMS)