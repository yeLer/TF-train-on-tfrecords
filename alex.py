# coding=utf-8
from __future__ import print_function
from tfrecords import read_tfrecords

import tensorflow as tf

learning_rate = 1e-3
training_iters = 20000
batch_size = 64
display_step = 20
save_step = 500

n_input = 784
n_classes = 10
dropout = 0.8

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
# tfrecords must be a list
train_data_dir = ['./tfrecords/train.tfrecords']
test_data_dir = ['./tfrecords/test.tfrecords']


def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'), b), name=name)


def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)


def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


def alex_net(_X, _weights, _biases, _dropout):
    _X = tf.reshape(_X, shape=[-1, 28, 28, 1])

    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])

    pool1 = max_pool('pool1', conv1, k=2)

    norm1 = norm('norm1', pool1, lsize=4)
    # Dropout
    norm1 = tf.nn.dropout(norm1, _dropout)

    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])

    pool2 = max_pool('pool2', conv2, k=2)

    norm2 = norm('norm2', pool2, lsize=4)
    # Dropout
    norm2 = tf.nn.dropout(norm2, _dropout)

    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])

    pool3 = max_pool('pool3', conv3, k=2)

    norm3 = norm('norm3', pool3, lsize=4)
    # Dropout
    norm3 = tf.nn.dropout(norm3, _dropout)

    dense1 = tf.reshape(norm3, [-1, _weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1')
    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2')  # Relu activation

    out = tf.matmul(dense2, _weights['out']) + _biases['out']
    return out


weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64])),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'wd1': tf.Variable(tf.random_normal([4 * 4 * 256, 1024])),
    'wd2': tf.Variable(tf.random_normal([1024, 1024])),
    'out': tf.Variable(tf.random_normal([1024, 10]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bc3': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'bd2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = alex_net(x, weights, biases, keep_prob)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# for train data prepare
batch_xs, batch_ys = read_tfrecords(train_data_dir, batch_size)
batch_xs_reshape = tf.reshape(batch_xs, [-1, n_input])
batch_ys_one_hot = tf.one_hot(batch_ys, depth=n_classes, axis=1)
# for test data prepare
batch_test_xs, batch_test_ys = read_tfrecords(test_data_dir, batch_size=256)
batch_test_xs = tf.reshape(batch_test_xs, [-1, n_input])
batch_ytest_one_hot = tf.one_hot(batch_test_ys, depth=n_classes, axis=1)
saver = tf.train.Saver(max_to_keep=5)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    epoches = 1
    # Keep training until reach max iterations
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print('*' * 20 + "get epoch 1" + "*" * 20)
    for step in range(training_iters+1):
        batch_xs_array, batch_ys_array = sess.run([batch_xs_reshape, batch_ys_one_hot])
        batch_xtest_array, batch_ytest_array = sess.run([batch_test_xs, batch_ytest_one_hot])
        old_epoches = step * batch_size//55000
        new_epoches = (step+1) * batch_size//55000
        if new_epoches !=old_epoches:
            print('*'*20+"get epoch %d"%(new_epoches+1)+"*"*20)
        sess.run(optimizer, feed_dict={x: batch_xs_array, y: batch_ys_array, keep_prob: dropout})
        if step % display_step == 0:
            acc_train, loss = sess.run([accuracy, cost], feed_dict={x: batch_xs_array, y: batch_ys_array, keep_prob: 1.})
            acc_test = sess.run(accuracy, feed_dict={x: batch_xtest_array, y: batch_ytest_array, keep_prob: 1.})
            print("Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss) +
                  ", Training Accuracy = " + "{:.5f}".format(acc_train)+"  Testing Accuracy:%g"%acc_test)
        step += 1
        if step % save_step == 0:
            saver.save(sess, save_path='./result/alex-mnist', global_step=step)
    coord.request_stop()
    coord.join(threads)

