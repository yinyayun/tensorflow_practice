'''
Created on 2018

@author: yinyayun
'''

import tensorflow as tf
import numpy as np
x2 = tf.constant(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))
with tf.Session() as sess:
    print(sess.run(tf.transpose(x2, perm=[0, 2, 1])))
    print(sess.run(tf.transpose(x2, perm=[1, 0, 2])))
