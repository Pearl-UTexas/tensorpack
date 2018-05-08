#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist-embeddings.py

import numpy as np
import argparse
import tensorflow as tf
import tensorflow.contrib.slim as slim
import random

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils.gpu import change_gpu

from spatial_relations_bb_cnn_data import get_test_data, DatasetPairs, DatasetTriplets

MATPLOTLIB_AVAIBLABLE = False
try:
    from matplotlib import offsetbox
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAIBLABLE = True
except ImportError:
    MATPLOTLIB_AVAIBLABLE = False

embed_dim = 2
opt = "SGD"
learning_rate = 1e-3


def contrastive_loss(left, right, y, margin, extra=False, scope="constrastive_loss"):
    r"""Loss for Siamese networks as described in the paper:
    `Learning a Similarity Metric Discriminatively, with Application to Face
    Verification <http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf>`_ by Chopra et al.

    .. math::
        \frac{1}{2} [y \cdot d^2 + (1-y) \cdot \max(0, m - d)^2], d = \Vert l - r \Vert_2

    Args:
        left (tf.Tensor): left feature vectors of shape [Batch, N].
        right (tf.Tensor): right feature vectors of shape [Batch, N].
        y (tf.Tensor): binary labels of shape [Batch]. 1: similar, 0: not similar.
        margin (float): horizon for negative examples (y==0).
        extra (bool): also return distances for pos and neg.

    Returns:
        tf.Tensor: constrastive_loss (averaged over the batch), (and optionally average_pos_dist, average_neg_dist)
    """
    with tf.name_scope(scope):
        y = tf.cast(y, tf.float32)

        delta = tf.reduce_sum(tf.square(left - right), 1)
        delta_sqrt = tf.sqrt(delta + 1e-10)

        match_loss = delta
        missmatch_loss = tf.square(tf.nn.relu(margin - delta_sqrt))

        loss = tf.reduce_mean(0.5 * (y * match_loss + (1 - y) * missmatch_loss))

        if extra:
            num_pos = tf.count_nonzero(y)
            num_neg = tf.count_nonzero(1 - y)
            pos_dist = tf.where(tf.equal(num_pos, 0), 0.,
                                tf.reduce_sum(y * delta_sqrt) / tf.cast(num_pos, tf.float32),
                                name="pos-dist")
            neg_dist = tf.where(tf.equal(num_neg, 0), 0.,
                                tf.reduce_sum((1 - y) * delta_sqrt) / tf.cast(num_neg, tf.float32),
                                name="neg-dist")
            return loss, pos_dist, neg_dist
        else:
            return loss


def siamese_cosine_loss(left, right, y, scope="cosine_loss"):
    r"""Loss for Siamese networks (cosine version).
    Same as :func:`contrastive_loss` but with different similarity measurement.

    .. math::
        [\frac{l \cdot r}{\lVert l\rVert \lVert r\rVert} - (2y-1)]^2

    Args:
        left (tf.Tensor): left feature vectors of shape [Batch, N].
        right (tf.Tensor): right feature vectors of shape [Batch, N].
        y (tf.Tensor): binary labels of shape [Batch]. 1: similar, 0: not similar.

    Returns:
        tf.Tensor: cosine-loss as a scalar tensor.
    """

    def l2_norm(t, eps=1e-12):
        """
        Returns:
            tf.Tensor: norm of 2D input tensor on axis 1
        """
        with tf.name_scope("l2_norm"):
            return tf.sqrt(tf.reduce_sum(tf.square(t), 1) + eps)

    with tf.name_scope(scope):
        y = 2 * tf.cast(y, tf.float32) - 1
        pred = tf.reduce_sum(left * right, 1) / (l2_norm(left) * l2_norm(right) + 1e-10)

        return tf.nn.l2_loss(y - pred) / tf.cast(tf.shape(left)[0], tf.float32)


def triplet_loss(anchor, positive, negative, margin, extra=False, scope="triplet_loss"):
    r"""Loss for Triplet networks as described in the paper:
    `FaceNet: A Unified Embedding for Face Recognition and Clustering
    <https://arxiv.org/abs/1503.03832>`_
    by Schroff et al.

    Learn embeddings from an anchor point and a similar input (positive) as
    well as a not-similar input (negative).
    Intuitively, a matching pair (anchor, positive) should have a smaller relative distance
    than a non-matching pair (anchor, negative).

    .. math::
        \max(0, m + \Vert a-p\Vert^2 - \Vert a-n\Vert^2)

    Args:
        anchor (tf.Tensor): anchor feature vectors of shape [Batch, N].
        positive (tf.Tensor): features of positive match of the same shape.
        negative (tf.Tensor): features of negative match of the same shape.
        margin (float): horizon for negative examples
        extra (bool): also return distances for pos and neg.

    Returns:
        tf.Tensor: triplet-loss as scalar (and optionally average_pos_dist, average_neg_dist)
    """

    with tf.name_scope(scope):
        d_pos = tf.reduce_sum(tf.square(anchor - positive), 1)
        d_neg = tf.reduce_sum(tf.square(anchor - negative), 1)

        loss = tf.reduce_mean(tf.maximum(0., margin + d_pos - d_neg))

        if extra:
            pos_dist = tf.reduce_mean(tf.sqrt(d_pos + 1e-10), name='pos-dist')
            neg_dist = tf.reduce_mean(tf.sqrt(d_neg + 1e-10), name='neg-dist')
            return loss, pos_dist, neg_dist
        else:
            return loss


def soft_triplet_loss(anchor, positive, negative, extra=True, scope="soft_triplet_loss"):
    r"""Loss for triplet networks as described in the paper:
    `Deep Metric Learning using Triplet Network
    <https://arxiv.org/abs/1412.6622>`_ by Hoffer et al.

    It is a softmax loss using :math:`(anchor-positive)^2` and
    :math:`(anchor-negative)^2` as logits.

    Args:
        anchor (tf.Tensor): anchor feature vectors of shape [Batch, N].
        positive (tf.Tensor): features of positive match of the same shape.
        negative (tf.Tensor): features of negative match of the same shape.
        extra (bool): also return distances for pos and neg.

    Returns:
        tf.Tensor: triplet-loss as scalar (and optionally average_pos_dist, average_neg_dist)
    """

    eps = 1e-10
    with tf.name_scope(scope):
        d_pos = tf.sqrt(tf.reduce_sum(tf.square(anchor - positive), 1) + eps)
        d_neg = tf.sqrt(tf.reduce_sum(tf.square(anchor - negative), 1) + eps)

        logits = tf.stack([d_pos, d_neg], axis=1)
        ones = tf.ones_like(tf.squeeze(d_pos), dtype="int32")

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=ones))

        if extra:
            pos_dist = tf.reduce_mean(d_pos, name='pos-dist')
            neg_dist = tf.reduce_mean(d_neg, name='neg-dist')
            return loss, pos_dist, neg_dist
        else:
            return loss


def center_loss(embedding, label, num_classes, alpha=0.1, scope="center_loss"):
    r"""Center-Loss as described in the paper
    `A Discriminative Feature Learning Approach for Deep Face Recognition`
    <http://ydwen.github.io/papers/WenECCV16.pdf> by Wen et al.

    Args:
        embedding (tf.Tensor): features produced by the network
        label (tf.Tensor): ground-truth label for each feature
        num_classes (int): number of different classes
        alpha (float): learning rate for updating the centers

    Returns:
        tf.Tensor: center loss
    """
    nrof_features = embedding.get_shape()[1]
    centers = tf.get_variable('centers', [num_classes, nrof_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alpha) * (centers_batch - embedding)
    centers = tf.scatter_sub(centers, label, diff)
    loss = tf.reduce_mean(tf.square(embedding - centers_batch), name=scope)
    return loss


class EmbeddingModel(ModelDesc):
    global embed_dim
    global opt
    def embed(self, x, nfeatures=embed_dim):
        """Embed all given tensors into an nfeatures-dim space.  """
        list_split = 0
        if isinstance(x, list):
            list_split = len(x)
            x = tf.concat(x, 0)

        # pre-process MNIST dataflow data
        #x = tf.expand_dims(x, 3)
        #x = x * 2 - 1

        # the embedding network
        net = slim.layers.fully_connected(x, 32, scope='fc1')
        net = slim.layers.fully_connected(net, 16, scope='fc2')
        embeddings = slim.layers.fully_connected(net, nfeatures, activation_fn=None, scope='fc3')

        # if "x" was a list of tensors, then split the embeddings
        if list_split > 0:
            embeddings = tf.split(embeddings, list_split, 0)

        return embeddings

    def optimizer(self):
        global learning_rate
        lr = tf.get_variable('learning_rate', initializer=learning_rate, trainable=False)
        if opt=='SGD':
            return tf.train.GradientDescentOptimizer(lr)
        elif opt=='Adam':
            return tf.train.AdamOptimizer(lr)
        elif opt=='Momentum':
            return tf.train.MomentumOptimizer(lr)
        elif opt=='RMSProp':
            return tf.train.RMSPropOptimizer(lr, momentum=0.9)


class SiameseModel(EmbeddingModel):
    @staticmethod
    def get_data():
        ds = DatasetPairs('data/amt_train.json','train')
        ds = BatchData(ds, 128 // 2) # mini batch gradient descent
        return ds

    def inputs(self):
        return [tf.placeholder(tf.float32, (None, 4112), 'input'),
                tf.placeholder(tf.float32, (None, 4112), 'input_y'),
                tf.placeholder(tf.int32, (None,), 'label')]

    def build_graph(self, x, y, label):
        global embed_dim
        # embed them
        single_input = x
        x, y = self.embed([x, y], embed_dim)

        # tag the embedding of 'input' with name 'emb', just for inference later on
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tf.identity(self.embed(single_input, embed_dim), name="emb")

        # compute the actual loss
        cost, pos_dist, neg_dist = contrastive_loss(x, y, label, 5., extra=True, scope="loss")
        cost = tf.identity(cost, name="cost")

        # track these values during training
        add_moving_summary(pos_dist, neg_dist, cost)
        return cost


class CosineModel(SiameseModel):
    def build_graph(self, x, y, label):
        global embed_dim
        single_input = x
        x, y = self.embed([x, y], embed_dim)

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tf.identity(self.embed(single_input, embed_dim), name="emb")

        cost = siamese_cosine_loss(x, y, label, scope="loss")
        cost = tf.identity(cost, name="cost")
        add_moving_summary(cost)
        return cost


class TripletModel(EmbeddingModel):
    @staticmethod
    def get_data():
        ds = DatasetTriplets('data/amt_train.json','train')       
        """
        s=0
	    for i in range(12):
	    print(len(ds.data_dict[i]))
	    s = s + len(ds.data_dict[i])
	    print(s)
        """
        ds = BatchData(ds, 128 // 3)    #128
        return ds

    def inputs(self):
        return [tf.placeholder(tf.float32, (None, 4112), 'input'),
                tf.placeholder(tf.float32, (None, 4112), 'input_p'),
                tf.placeholder(tf.float32, (None, 4112), 'input_n'),
                ]

    def loss(self, a, p, n):
        return triplet_loss(a, p, n, 5., extra=True, scope="loss")

    def build_graph(self, a, p, n):
        global embed_dim
        single_input = a
        a, p, n = self.embed([a, p, n], embed_dim)

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tf.identity(self.embed(single_input, embed_dim), name="emb")

        cost, pos_dist, neg_dist = self.loss(a, p, n)

        cost = tf.identity(cost, name="cost")
        add_moving_summary(pos_dist, neg_dist, cost)
        return cost


class SoftTripletModel(TripletModel):
    def loss(self, a, p, n):
        return soft_triplet_loss(a, p, n, scope="loss")


class CenterModel(EmbeddingModel):
    @staticmethod
    def get_data():
        #ds = dataset.Mnist('train')
        ds = Dataset('data/amt_train.json','train')    
        ds = BatchData(ds, 128)
        return ds

    def inputs(self):
        return [tf.placeholder(tf.float32, (None, 4112), 'input'),
                tf.placeholder(tf.int32, (None,), 'label')]

    def build_graph(self, x, label):
        global embed_dim
        # embed them
        x = self.embed(x, embed_dim)
        x = tf.identity(x, name='emb')

        # compute the embedding loss
        emb_cost = center_loss(x, label, 10, 0.01)
        # compute the classification loss
        logits = slim.layers.fully_connected(tf.nn.relu(x), 10, activation_fn=None, scope='logits')

        cls_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label),
                                  name='classification_costs')
        total_cost = tf.add(emb_cost, 100 * cls_cost, name="cost")

        # track these values during training
        add_moving_summary(total_cost, cls_cost, emb_cost)
        return total_cost


def get_config(model, algorithm_name):

    extra_display = ["cost"]
    if not algorithm_name == "cosine" and not algorithm_name == "center":
        extra_display = extra_display + ["loss/pos-dist", "loss/neg-dist"]

    return TrainConfig(
        dataflow=model.get_data(),
        model=model(),
        callbacks=[
            #ModelSaver(),
            ModelSaver(max_to_keep=20, keep_checkpoint_every_n_hours=2)
            #ScheduledHyperParamSetter('learning_rate', [(10, 1e-5), (20, 1e-6)])
        ],
        extra_callbacks=[
            MovingAverageSummary(),
            ProgressBar(extra_display),
            MergeAllSummaries(),
            RunUpdateOps()
        ],
        max_epoch=10000,
    )


def visualize(model_path, model, algo_name):
    if not MATPLOTLIB_AVAIBLABLE:
        logger.error("visualize requires matplotlib package ...")
        return
    pred = OfflinePredictor(PredictConfig(
        session_init=get_model_loader(model_path),
        model=model(),
        input_names=['input','bb'],
        output_names=['emb']))

    NUM_BATCHES = 6
    BATCH_SIZE = 128
    #images = np.zeros((BATCH_SIZE * NUM_BATCHES, 28, 28))  # the used digits
    embed = np.zeros((BATCH_SIZE * NUM_BATCHES, 2))  # the actual embeddings in 2-d
    labels = np.zeros((BATCH_SIZE * NUM_BATCHES))

    # get only the embedding model data (MNIST test)
    ds = get_test_data('data/amt_test.json')
    ds.reset_state()

    for offset, dp in enumerate(ds.get_data()):
        data, label = dp
        prediction = pred(data)[0]
	print(prediction)
        embed[offset * BATCH_SIZE:offset * BATCH_SIZE + BATCH_SIZE, ...] = prediction
        labels[offset * BATCH_SIZE:offset * BATCH_SIZE + BATCH_SIZE, ...] = label
        offset += 1
        if offset == NUM_BATCHES:
            break

    plt.figure()
    ax = plt.subplot(111)
    ax_min = np.min(embed, 0)
    ax_max = np.max(embed, 0)

    ax_dist_sq = np.sum((ax_max - ax_min)**2)
    ax.axis('off')

    relation_labels = { 
        0:'on',
        1:'in',
        2:'near',
        3:'beside',
        4:'next to',
        5:'to the left of',
        6:'to the right of',
        7:'below',
        8:'above',
        9:'at',
        10:'behind',
        11:'on top of'
    }
    circles = []
    classes = []

    # total number of labels
    N = 12
    # define the colormap
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    x = np.arange(N)
    ys = [i+x+(i*x)**2 for i in range(N)]
    #c = cm.rainbow(np.linspace(0, 1, len(ys)))
    c = ['r','b', 'c', 'g','yellow','blueviolet','lightblue','darkgreen','orange','mediumvioletred','lightcoral',
            'olive']#,'brown','dimgray','steelblue','k']

    for i in relation_labels:
        circles.append(mpatches.Circle((0,0),1,color=c[i]))
        classes.append(relation_labels[i])

    shown_images = np.array([[1., 1.]])
    for i in range(embed.shape[0]):
        dist = np.sum((embed[i] - shown_images)**2, 1)
        if np.min(dist) < 3e-4 * ax_dist_sq:     # don't show points that are too close
            continue
        shown_images = np.r_[shown_images, [embed[i]]]
        #imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(np.reshape(images[i, ...], [28, 28]),
        #                                    zoom=0.6, cmap=plt.cm.gray_r), xy=embed[i], frameon=False)
        #ax.add_artist(imagebox)
        plt.scatter(embed[i][0], embed[i][1], color=c[int(labels[i])])

    plt.axis([ax_min[0], ax_max[0], ax_min[1], ax_max[1]])
    plt.xticks([]), plt.yticks([])
    plt.title('Embedding using %s-loss' % algo_name)
    plt.savefig('%s.jpg' % algo_name)

def evaluate_random(model_path, model, algo_name):
    global embed_dim
    ensemble_size = 15
    correct = 0
    total = 0
    BATCH_SIZE = 128
    #NUM_BATCHES = 50000

    pred = OfflinePredictor(PredictConfig(
            session_init=get_model_loader(model_path),
            model=model(),
            input_names=['input'],
            output_names=['emb']))

    # get train data
    dt = get_test_data('data/amt_train.json', 'train')
    dt = BatchData(dt, BATCH_SIZE)
    #dt = get_test_data()
    dt.reset_state()
    print('loaded training data')

    train_data = {}
    for offset,dp in enumerate(dt.get_data()):
        #print(offset)
        data, label = dp
        prediction = pred(data)
        embedding = prediction[0]
        for i in range(BATCH_SIZE):
            gt = label[i]
            if gt not in train_data:
                train_data[gt] = [embedding[i]]
            else:
                train_data[gt].append(embedding[i])
        offset += 1
        #if offset == NUM_BATCHES:
        #    break

    total_tr_data = 0
    for label in train_data:
        print(str(label) + ': '+ str(len(train_data[label])))
        total_tr_data += len(train_data[label])
    print('total training data: ' + str(total_tr_data))

    ds = get_test_data('data/amt_test.json')
    ds.reset_state()
    print('loaded test data')

    for dp in ds.get_data():
        data, label = dp
        embed_test_batch = pred(data)[0]
        dist = {}
        for i in range(BATCH_SIZE):
            embed_test = embed_test_batch[i]
            # choose an image randomly from every class
            for l in train_data:
                dist[l] = 0
                r = random.sample(range(0,len(train_data[l])), ensemble_size)
                for sample in r:
                    dist[l] += np.linalg.norm(embed_test-train_data[l][sample])
                dist[l] = dist[l]/ensemble_size

            min_value = min(dist.itervalues())
            min_keys = [k for k in dist if dist[k] == min_value]
            if len(min_keys)==1:
                pred_class = min_keys[0]
            else:
                pred_class = min_keys[random.randint(0,len(min_keys)-1)]

            if pred_class == label[i]:
                correct += 1
            total += 1

    print('total test data: ' + str(total))
    return correct, total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('-a', '--algorithm', help='used algorithm', required=True,
                        choices=["siamese", "cosine", "triplet", "softtriplet", "center"])
    parser.add_argument('--visualize', help='export embeddings into an image', action='store_true')
    parser.add_argument('--evaluate', help = 'compute accuracy', action='store_true')
    parser.add_argument('--dim', help='dimensionality of the embedding space', type=int)
    parser.add_argument('--modelname', help = 'model directory name', type=str)
    parser.add_argument('--opt', help = 'Optimizer', type=str)
    parser.add_argument('--lr', help='learning rate', type=float)
    args = parser.parse_args()

    ALGO_CONFIGS = {"siamese": SiameseModel,
                    "cosine": CosineModel,
                    "triplet": TripletModel,
                    "softtriplet": SoftTripletModel,
                    "center": CenterModel}

    if args.modelname:
        logger.auto_set_dir(name=args.modelname)
    else:
        logger.auto_set_dir(name=args.algorithm)

    with change_gpu(args.gpu):
        if args.visualize:
            visualize(args.load, ALGO_CONFIGS[args.algorithm], args.algorithm)
	elif args.evaluate:
            correct, total = evaluate_random(args.load, ALGO_CONFIGS[args.algorithm], args.algorithm)
            print('accuracy: '+str(float(correct)*100/total) + '% = ' + str(correct) + '/' +str(total))

        else:
            config = get_config(ALGO_CONFIGS[args.algorithm], args.algorithm)
            if args.load:
                config.session_init = SaverRestore(args.load)
            else:
                launch_train_with_config(config, SimpleTrainer())