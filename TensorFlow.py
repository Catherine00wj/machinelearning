## Submission.py for COMP6714-Project2
###################################################################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import zipfile
import spacy
import numpy as np
import collections
import math
from tempfile import gettempdir
import tensorflow as tf
import gensim
import re
import random
import pdb
from gensim.models import KeyedVectors
from six.moves import urllib
from six.moves import range
from six.moves.urllib.request import urlretrieve
from gensim.models.keyedvectors import KeyedVectors
import math

import pickle

total_list = []
dictionary_fre = {}
dictionary = {}
reverse_dictionary = {}
data = []
count = []
ccc = []
labels = []
label = []
vocabulary_size = 10000
skip_window = 1  # How many words to consider left and right of the target word.
num_samples = 2
#--------------------------------------------------------------------------------------------------------
graph = tf.Graph()
def adjective_embeddings(data_file, embeddings_file_name, num_steps, embedding_dim):
    # Specification of Training data:

    batch_size = 64  # Size of mini-batch for skip-gram model.
    embedding_size = embedding_dim  # Dimension of the embedding vector.
     # How many times to reuse an input to generate a label.
    num_sampled = 200  # Sample size for negative examples.
    logs_path = './log/'
    learning_rate_ = 0.01
    # Specification of test Sample:
    sample_size = 20  # Random sample of words to evaluate similarity.
    sample_window = 20  # Only pick samples in the head of the distribution.
    sample_examples = np.random.choice(sample_window, sample_size, replace=False)  # Randomly pick a sample of size 16

    f = open(data_file, 'rb')
    dictionary , reverse_dictionary,read_data,read_label  = pickle.load(f)
    print("ddddddd", reverse_dictionary)
    print()
    print("rrrrrrrr", read_label)
    print("wwwwwwww", read_data)
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    ## Constructing the graph...
    graph = tf.Graph()
    with graph.as_default():
        with tf.device('/cpu:0'):
            # Placeholders to read input data.
            with tf.name_scope('Inputs'):
                train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
                train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

            # Look up embeddings for inputs.
            with tf.name_scope('Embeddings'):
                sample_dataset = tf.constant(sample_examples, dtype=tf.int32)
                embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)

                # Construct the variables for the NCE loss
                nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                              stddev=1.0 / math.sqrt(embedding_size)))
                nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            with tf.name_scope('Loss'):
                loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
                                                     labels=train_labels, inputs=embed,
                                                     num_sampled=num_sampled, num_classes=vocabulary_size))

            # Construct the Gradient Descent optimizer using a learning rate of 0.01.
            with tf.name_scope('Adam_Optimizer'):
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_).minimize(loss)

            # Normalize the embeddings to avoid overfitting.
            with tf.name_scope('Normalization'):
                norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
                normalized_embeddings = embeddings / norm

            sample_embeddings = tf.nn.embedding_lookup(normalized_embeddings, sample_dataset)
            similarity = tf.matmul(sample_embeddings, normalized_embeddings, transpose_b=True)

            # Add variable initializer.
            init = tf.global_variables_initializer()

            # Create a summary to monitor cost tensor
            tf.summary.scalar("cost", loss)
            # Merge all summary variables.
            merged_summary_op = tf.summary.merge_all()

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        session.run(init)
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        print('Initializing the model')
        length = len(read_data)
        average_loss = 0
        for step in range(num_steps):
            print(step)
            batch_inputs = np.ndarray(shape=(batch_size), dtype=np.int32)
            batch_labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
            aa = step * batch_size % length
            bb = 0
            for bb in range(batch_size):
                batch_inputs[bb] = read_data[aa]
                batch_labels[bb,0] = read_label[aa]
                aa = aa +1
                if aa == length:
                    aa =0
            # batch_inputs, batch_labels = train_inputs, train_labels
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # We perform one update step by evaluating the optimizer op using session.run()
            _, loss_val, summary = session.run([optimizer, loss, merged_summary_op], feed_dict=feed_dict)

            summary_writer.add_summary(summary, step)
            average_loss += loss_val

            if step % 5000 == 0:
                if step > 0:
                    average_loss /= 5000

                    # The average loss is an estimate of the loss over the last 5000 batches.
                    print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0

            # Evaluate similarity after every 10000 iterations.
            if step % 10000 == 0:
                sim = similarity.eval()  #
                for i in range(sample_size):
                    sample_word = reverse_dictionary[sample_examples[i]]
                    top_k = 10  # Look for top-10 neighbours for words in sample set.
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    print(top_k)
                    log_str = 'Nearest to %s:' % sample_word
                    for k in range(top_k):
                        print("22222222", nearest[k])
                        close_word = reverse_dictionary[nearest[k]]
                        # print("22222222", nearest[k])
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)
                print()
        final_embeddings = normalized_embeddings.eval()

        embedding_number = 0
        embedding_index_number_list = list()
        with open(embeddings_file_name, 'w') as outputfile:
            outputfile.write(str(len(final_embeddings)))
            outputfile.write(' ')
            outputfile.write(str(embedding_size))
            for i in range(len(final_embeddings)):
                outputfile.write('\n')
                outputfile.write(reverse_dictionary[i])
                for j in range(embedding_size):
                    outputfile.write(' ')
                    outputfile.write(str(round(final_embeddings[i][j],6)))

def process_data(input_data):
    vocabulary_size = 10000
    z = zipfile.ZipFile(input_data, "r")
    suffix = "txt"
    nlp = spacy.load("en")
    skip_window = 1
    num_samples = 2
    return_file =''
    # ---------initial process-----------------------------------------------------------------------------
    for filename in z.namelist():
        filename = str(filename)
        if filename.endswith(suffix):
            content = z.read(filename)
            document = nlp(content.decode('utf8'))
            for i in range(len(document)):
                if document[i].is_space:
                    continue
                # if document[i].string == "\n":
                #     continue
                if document[i].is_alpha:
                    if document[i].pos_ == 'ADJ':
                    # document[i].string.strip()
                        aa = document[i].string.lower().strip()
                        total_list.append(aa)
                if document[i].pos_ == 'NUM':
                    aa = 'n'
                    total_list.append(aa)
    # print(dictionary_total)

    for i in range(len(total_list)):
        if total_list[i] not in dictionary_fre.keys():
            dictionary_fre[total_list[i]] = 1
        elif total_list[i] in dictionary_fre.keys():
            dictionary_fre[total_list[i]] += 1
    # print(dictionary_fre)
    dic_sort = sorted(dictionary_fre.items(), key=lambda x: x[1], reverse=True)
    # print(dic_sort)
    # print("55555  ", len(dic_sort))
    # print(dic_sort)
    for j in range(len(dic_sort)):
        if j < vocabulary_size:
            dictionary[dic_sort[j][0]] = j
            reverse_dictionary[j] = dic_sort[j][0]
    # print("77777777777  " , len(dictionary))
    print(dictionary)
    # print("0000000000   ", len(reverse_dictionary))
    print(reverse_dictionary)

    # -----------data and lable-----------------------------------------------------
    z = zipfile.ZipFile(input_data, "r")
    for filename in z.namelist():
        filename = str(filename)
        if filename.endswith(suffix):
            # print("666666666")
            content = z.read(filename)
            document = nlp(content.decode('utf8'))
            for i in range(len(document)):
                # if document[i].string == "\n":
                #     continue
                if document[i].is_space:
                    continue
                if document[i].is_alpha and document[i].string.lower().strip() in dictionary:
                    ccc = document[i].string.lower().strip()
                    # print("66666666", dictionary[ccc])
                    # print(ccc)
                    # print("7777777")
                    aaa = document[i].children
                    child_count = 0
                    for j in aaa:
                        if j.string.lower().strip() in dictionary:
                            count.append(j.string.lower().strip())
                            child_count += 1
                    for m in range(skip_window):
                        b = i - (m + 1)
                        if document[b].is_alpha and document[b].string.lower().strip() in dictionary:
                            # print(1)
                            count.append(document[b].string.lower().strip())
                            child_count += 1
                        d = i + (m + 1)
                        if d <= len(document) - 1:
                            if document[d].is_alpha and document[d].string.lower().strip() in dictionary:
                                count.append(document[d].string.lower().strip())
                                child_count += 1
                    for s in range(num_samples):
                        if s >= len(count):
                            break
                        data.append(dictionary[ccc])
                        if dictionary[ccc] >3624:
                            print("4444444", dictionary[ccc])
                        # print(dictionary[ccc])
                        label.append(dictionary[count[s]])
                        if dictionary[count[s]] > 3624:
                            print(count[s])
                            print(dictionary[count[s]])
    # print("888888" , len(data))
    # print("99999999  ", len(label))
    print("81111111111",data)
    print()
    print("333333333", label)
    f = open("data.txt", "wb")
    pickle.dump((dictionary,reverse_dictionary,data,label) ,f)

    return_file = "data.txt"
    return return_file
def Compute_topk(model_file, input_adjective, top_k):
    word_vectors = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False)
    top_k_pair = word_vectors.most_similar(input_adjective, topn=top_k)
    top_k_list = []
    for i in range(top_k):
        top_k_list.append(top_k_pair[i][0])
    return top_k_list


process_data("BBC_Data.zip")
adjective_embeddings("data.txt", "adjective_embeddings.txt", 100001, 200)
# aaaaaaaa = Compute_topk("adjective_embeddings.txt", "less", 100)
# print(aaaaaaaa)
