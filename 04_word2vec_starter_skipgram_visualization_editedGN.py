from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from process_data import process_data

VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128 # dimension of the word embedding vectors
SKIP_WINDOW = 1 # the context window
NUM_SAMPLED = 64    # Number of negative examples to sample.
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 10000
SKIP_STEP = 2000 # how many steps to skip before reporting the loss

def word2vec(batch_gen):
    """ Build the graph for word2vec model and train it """
    # Step 1: define the placeholders for input and output
    # center_words have to be int to work on embedding lookup

    # TO DO
    with tf.name_scope('data'):
        center_word = tf.placeholder(tf.int32, shape = [BATCH_SIZE])
        target_words = tf.placeholder(tf.int32, shape = [BATCH_SIZE,1])

    # Step 2: define weights. In word2vec, it's actually the weights that we care about
    # vocab size x embed size
    # initialized to random uniform -1 to 1

    # TOO DO
    with tf.name_scope('embed'):
        embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0))

    # Step 3: define the inference
    # get the embed of input words using tf.nn.embedding_lookup
    # embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')

    # TO DO
    with tf.name_scope('loss'):
        embed = tf.nn.embedding_lookup(embed_matrix, center_word, name='embed')

    # Step 4: construct variables for NCE loss
    # tf.nn.nce_loss(weights, biases, labels, inputs, num_sampled, num_classes, ...)
    # nce_weight (vocab size x embed size), intialized to truncated_normal stddev=1.0 / (EMBED_SIZE ** 0.5)
    # bias: vocab size, initialized to 0

    # TO DO
        nce_bias = tf.zeros(shape=[VOCAB_SIZE], name='nce_bias')
        nce_weight = tf.Variable(tf.truncated_normal(shape=[VOCAB_SIZE, EMBED_SIZE], stddev=1.0/EMBED_SIZE**0.5), name='nce_weight')
    
    
    # define loss function to be NCE loss function
    # tf.nn.nce_loss(weights, biases, labels, inputs, num_sampled, num_classes, ...)
    # need to get the mean accross the batch

    # TO DO
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, 
                                         biases=nce_bias, 
                                         labels=target_words, 
                                         inputs=embed, 
                                         num_sampled=NUM_SAMPLED, 
                                         num_classes=VOCAB_SIZE), name='loss')

    # Step 5: define optimizer
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
    
    # TO DO

    with tf.Session() as sess:
        # TO DO: initialize variables
        sess.run(tf.global_variables_initializer())

        total_loss = 0.0 # we use this to calculate the average loss in the last SKIP_STEP steps
        writer = tf.summary.FileWriter('./my_graph/no_frills_namescope/', sess.graph)
        
        for index in xrange(NUM_TRAIN_STEPS):
            centers, targets = batch_gen.next()
            # TO DO: create feed_dict, run optimizer, fetch loss_batch
            pred, loss_batch = sess.run([optimizer, loss], feed_dict={center_word:centers, target_words:targets})

            total_loss += loss_batch
            if (index + 1) % SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
                total_loss = 0.0
        writer.close()
        
                ####################
        #code to visualize the embeddings. uncomment the below to visualize embeddings
        final_embed_matrix = sess.run(embed_matrix)
        
        # it has to variable. constants don't work here. you can't reuse model.embed_matrix
        embedding_var = tf.Variable(final_embed_matrix[:1000], name='embedding')
        sess.run(embedding_var.initializer)

        config = projector.ProjectorConfig()
        summary_writer = tf.summary.FileWriter('processed')

        # add embedding to the config file
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
    
        # link this tensor to its metadata file, in this case the first 500 words of vocab
        embedding.metadata_path = './my_graph/no_frills_namescope/vocab_1000.tsv'

        # saves a configuration file that TensorBoard will read during startup.
        projector.visualize_embeddings(summary_writer, config)
        saver_embed = tf.train.Saver([embedding_var])
        #saver_embed.save(sess, 'processed/model3.ckpt', 1)
        saver_embed.save(sess, './my_graph/no_frills_namescope/model3.ckpt', 1)

def main():
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    word2vec(batch_gen)

if __name__ == '__main__':
    main()