#!/usr/bin/env python
# encoding: utf-8
'''
@author: lele Ye
@contact: 1750112338@qq.com
@software: pycharm 2018.2
@file: tfrecords.py
@time: 2019/5/6 15:43
@desc:
'''
# coding:utf-8
# 将MNIST输入数据转化为TFRecord的格式

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from PIL import Image
import os
from tqdm import tqdm


# 把传入的value转化为整数型的属性，int64_list对应着 tf.train.Example 的定义
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 把传入的value转化为字符串型的属性，bytes_list对应着 tf.train.Example 的定义
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_mnsit_tfreords(source_dir, name="train"):
    # 读取MNIST数据
    mnist = input_data.read_data_sets(source_dir, dtype=tf.uint8, one_hot=True)
    images = filename = labels = num_examples = None
    if name == "train":
        # 训练数据的图像，可以作为一个属性来存储
        images = mnist.train.images
        # 训练数据所对应的正确答案，可以作为一个属性来存储
        labels = mnist.train.labels
        filename = "./tfrecords/train.tfrecords"
        # 训练数据的图像分辨率，可以作为一个属性来存储
        pixels = images.shape[0]
        # 训练数据的个数
        num_examples = mnist.train.num_examples
    elif name == "test":
        # test数据的图像，可以作为一个属性来存储
        images = mnist.test.images
        # test数据所对应的正确答案，可以作为一个属性来存储
        labels = mnist.test.labels
        filename = "./tfrecords/test.tfrecords"
        # 训练数据的个数
        num_examples = mnist.test.num_examples
    # 指定要写入TFRecord文件的地址
    if not os.path.exists("./tfrecords"):
        os.mkdir("./tfrecords")
    # 创建一个write来写TFRecord文件
    writer = tf.python_io.TFRecordWriter(filename)
    for index in tqdm(range(num_examples)):
        # 把图像矩阵转化为字符串
        image_raw = images[index].tostring()
        # 将一个样例转化为Example Protocol Buffer，并将所有的信息写入这个数据结构
        example = tf.train.Example(features=tf.train.Features(feature={
            # 'pixels': _int64_feature(pixels),
            'label': _int64_feature(np.argmax(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        # 将 Example 写入TFRecord文件
        writer.write(example.SerializeToString())

    writer.close()

# 读取TFRecord文件中的数据
def read_tfrecords(data_dir, batch_size):
    # 创建一个reader来读取TFRecord文件中的样例
    reader = tf.TFRecordReader()
    # 通过 tf.train.string_input_producer 创建输入队列
    filename_queue = tf.train.string_input_producer(data_dir)
    # 从文件中读取一个样例
    _, serialized_example = reader.read(filename_queue)
    # 解析读入的一个样例
    features = tf.parse_single_example(
        serialized_example,
        features={
            # 这里解析数据的格式需要和上面程序写入数据的格式一致
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    # tf.decode_raw可以将字符串解析成图像对应的像素数组
    images = tf.decode_raw(features['image_raw'], tf.uint8)
    images = tf.reshape(images, [28, 28, 1])

    # tf.cast可以将传入的数据转化为想要改成的数据类型
    labels = tf.cast(features['label'], tf.int32)
    num_preprocess_threads = 12
    min_queue_examples = 50
    images_batch, label_batch = tf.train.shuffle_batch(
        [images, labels],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
    return images_batch, label_batch


def read_to_test(images_batch, label_batch, batch_size):
    image = tf.reshape(images_batch, [batch_size, 28, 28])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # 每次运行可以读取TFRecord文件中的一个样例。当所有样例都读完之后，再次样例中的程序会重头读取
        for i in range(batch_size):
            data, label = sess.run([image[i, :, :], label_batch[i]])
            result = Image.fromarray(data)
            result.save(str(i) + '.png')
            print(label)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    # create_mnsit_tfreords(source_dir="./mnist_data", name="train")
    # create_mnsit_tfreords(source_dir="./mnist_data", name="test")
    batch_size = 8
    data_dir = ["./tfrecords/train.tfrecords"]
    images_batch, label_batch = read_tfrecords(data_dir, batch_size)
    read_to_test(images_batch, label_batch, batch_size)
