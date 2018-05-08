import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import vgg 
import numpy as np
import cv2

slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

sess = tf.Session('', tf.Graph())
with sess.graph.as_default():
    saver = tf.train.Saver()
    saver.restore(sess, "/media/internal/asaran/vgg_16.ckpt")

    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    img1 = imread('laska.png', mode='RGB')
    img1 = imresize(img1, (224, 224))
    
    #coord = tf.train.Coordinator()
    #threads = []
    #for qr in sess.graph.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
    #    threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
    #                                     start=True))
    #logits = sess.run('softmax_linear/softmax_linear:0', 
    #                 feed_dict={'is_training:0': False, 'imgs:0': img1})
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, end_points = vgg.vgg_16([img1], num_classes=FLAGS.num_classes)


    print(logits)
    #saver.restore(sess, ckpt.model_checkpoint_path)
    #softmaxval = sess.run(vgg.vgg_16)
    #tf.get_default_graph().get_tensor_by_name("VGG16/fc16:0")
