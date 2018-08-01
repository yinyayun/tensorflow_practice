'''
Created on 2018��7��31��

@author: yinyayun
'''

import tensorflow as tf
import  numpy as np
from sklearn.cross_validation import train_test_split
from data.data_utils import loadZip

max_seq_len = 10
word_embeding_size = 100
rnn_size = 28
lrate = 0.01
epochs = 50
batch_size = 128

def load_data(texts, targets, test_size=0):
    # 构建词典并向量化
    vocab = tf.contrib.learn.preprocessing.VocabularyProcessor(max_seq_len, min_frequency=0)
    texts_process = np.array(list(vocab.fit_transform(texts)))
    vocab_size = len(vocab.vocabulary_)
    # 打乱顺序
    shuffle_x = np.random.permutation(np.arange(len(targets)))
    x_shuffle = texts_process[shuffle_x]
    y_shuffle = np.array(targets)[shuffle_x]
    # 分割数据集
    x_train, x_test, y_train, y_test = train_test_split(x_shuffle, y_shuffle, test_size=test_size)
    return vocab_size, x_train, x_test, y_train, y_test


def build_model(vocab_size, x_input, y_output, keep_prob):
    # embedding
    embedding_weights = tf.Variable(tf.random_uniform([vocab_size, word_embeding_size], -1., 1.))
    embeding_output = tf.nn.embedding_lookup(embedding_weights, x_input)
    # RNN动态序列
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=rnn_size)
    output, _ = tf.nn.dynamic_rnn(cell, embeding_output, dtype=tf.float32)
    output = tf.nn.dropout(output, keep_prob)
    #
    output = tf.transpose(output, [1, 0, 2])
    last = tf.gather(output, int(output.get_shape()[0]) - 1)
    output = tf.transpose(output, [1, 0, 2])
    # RNN之后增加全连接层
    weight = tf.Variable(tf.truncated_normal([rnn_size, 2], stddev=0.1))
    bias = tf.Variable(tf.constant(0.1, shape=[2]))
    # 定义loss损失
    logits = tf.nn.softmax(tf.matmul(last, weight) + bias)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_output)
    loss = tf.reduce_mean(losses)
    # 优化器
    optimizer = tf.train.RMSPropOptimizer(lrate)
    train_step = optimizer.minimize(loss)
    # 准确率
    output_index = tf.argmax(logits, 1)
    acc = tf.reduce_mean(tf.cast(tf.equal(output_index, tf.cast(y_output, tf.int64)), tf.float32))
    return train_step, loss, acc

def train(x_train, y_train, x_test, y_test, vocab_size):
    x_input = tf.placeholder(tf.int32, [None, max_seq_len])
    y_output = tf.placeholder(tf.int32, [None])
    keep_prob = tf.placeholder(tf.float32)
    train_step, loss, acc = build_model(vocab_size, x_input, y_output, keep_prob)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        num_batchs = int(len(x_train) / batch_size) + 1
        print('start train, batchs:', num_batchs)
        for epoch in range(epochs):
            for i in range(num_batchs):
                s_index = i * batch_size
                e_index = np.min([len(x_train), (i + 1) * batch_size])
                batch_train = x_train[s_index:e_index]
                batch_output = y_train[s_index:e_index]
                feed = {
                    x_input:batch_train,
                    y_output:batch_output,
                    keep_prob:0.05
                    }
                sess.run(train_step, feed_dict=feed)
                if(i % 200 == 0):
                    train_loss, train_acc = sess.run([loss, acc], feed_dict=feed)
                    feed = {
                        x_input:x_test,
                        y_output:y_test,
                        keep_prob:0.05
                    }
                    test_loss, test_acc = sess.run([loss, acc], feed_dict=feed)
                    print("epoch:{},batch:{}/{}, train_loss:{:.2},train_acc:{:.2},test_loss:{:.2},test_acc:{:.2}"
                          .format(epoch, i, num_batchs, train_loss, train_acc, test_loss, test_acc))
if __name__ == '__main__':
    texts, targets = loadZip('../data/toutiao_cat_data_tokens.zip')
    vocab_size, x_train, x_test, y_train, y_test = load_data(texts, targets, 0.1)
    train(x_train, y_train, x_test, y_test, vocab_size)
    
