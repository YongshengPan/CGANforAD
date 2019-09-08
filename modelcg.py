# Basic Code is taken from https://github.com/ckmarkoh/GAN-tensorflow

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from scipy.misc import imsave
import os
import shutil
from PIL import Image
import time
import random

from layers import *

img_layer = 1


def build_resnet_block(inputres, dim, name="resnet"):
    with tf.variable_scope(name):
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = general_conv3d(out_res, dim, 3, 3, 3, 1, 1, 1, 0.02, "VALID", "c1")
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = general_conv3d(out_res, dim, 3, 3, 3, 1, 1, 1, 0.02, "VALID", "c2", do_relu=False)

        return tf.nn.relu(out_res + inputres)


def build_generator_resnet_0blocks(inputgen, dim, name="generator"):
    with tf.variable_scope(name):
        f = 7
        ks = 3

        pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
        o_c1 = general_conv3d(pad_input, dim, f, f, f, 1, 1, 1, 0.02, name="c1")
        o_c2 = general_conv3d(o_c1, dim * 2, ks, ks, ks, 2, 2, 2, 0.02, "SAME", "c2")
        o_c3 = general_conv3d(o_c2, dim * 4, ks, ks, ks, 2, 2, 2, 0.02, "SAME", "c3")

        o_c4 = general_deconv3d(o_c3, [1, 64, 64, 64, dim * 2], dim * 2, ks, ks, ks, 2, 2, 2, 0.02, "SAME", "c4")
        o_c5 = general_deconv3d(o_c4, [1, 128, 128, 128, dim], dim, ks, ks, ks, 2, 2, 2, 0.02, "SAME", "c5")
        o_c5_pad = tf.pad(o_c5, [[0, 0], [ks, ks], [ks, ks],[ks, ks], [0, 0]], "REFLECT")
        o_c6 = general_conv3d(o_c5_pad, img_layer, f, f, f, 1, 1, 1, 0.02, "VALID", "c6", do_relu=False)

        out_gen = tf.nn.relu(o_c6, "t1")-1
        return out_gen


def build_generator_resnet_3blocks(inputgen, dim, name="generator"):
    with tf.variable_scope(name):
        f = 7
        ks = 3

        pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
        o_c1 = general_conv3d(pad_input, dim, f, f, f, 1, 1, 1, 0.02, name="c1")
        o_c2 = general_conv3d(o_c1, dim * 2, ks, ks, ks, 2, 2, 2, 0.02, "SAME", "c2")
        o_c3 = general_conv3d(o_c2, dim * 4, ks, ks, ks, 2, 2, 2, 0.02, "SAME", "c3")

        o_r1 = build_resnet_block(o_c3, dim * 4, "r1")
        o_r2 = build_resnet_block(o_r1, dim * 4, "r2")
        o_r3 = build_resnet_block(o_r2, dim * 4, "r3")

        o_c4 = general_deconv3d(o_r3, [1, 64, 64, 64, dim * 2], dim * 2, ks, ks, ks, 2, 2, 2, 0.02, "SAME", "c4")
        o_c5 = general_deconv3d(o_c4, [1, 128, 128, 128, dim], dim, ks, ks, ks, 2, 2, 2, 0.02, "SAME", "c5")
        o_c5_pad = tf.pad(o_c5, [[0, 0], [ks, ks], [ks, ks],[ks, ks], [0, 0]], "REFLECT")
        o_c6 = general_conv3d(o_c5_pad, img_layer, f, f, f, 1, 1, 1, 0.02, "VALID", "c6", do_relu=False)

        out_gen = tf.nn.relu(o_c6/2+1, "t1") - 1
        return out_gen


def build_generator_resnet_6blocks(inputgen, dim, name="generator"):
    with tf.variable_scope(name):
        f = 7
        ks = 3
        pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
        o_c1 = general_conv3d(pad_input, dim, f, f, f, 1, 1, 1, 0.02, name="c1")
        o_c2 = general_conv3d(o_c1, dim * 2, ks, ks, ks, 2, 2, 2, 0.02, "SAME", "c2")
        o_c3 = general_conv3d(o_c2, dim * 4, ks, ks, ks, 2, 2, 2, 0.02, "SAME", "c3")
        o_r1 = build_resnet_block(o_c3, dim * 4, "r1")
        o_r2 = build_resnet_block(o_r1, dim * 4, "r2")
        o_r3 = build_resnet_block(o_r2, dim * 4, "r3")
        o_r4 = build_resnet_block(o_r3, dim * 4, "r4")
        o_r5 = build_resnet_block(o_r4, dim * 4, "r5")
        o_r6 = build_resnet_block(o_r5, dim * 4, "r6")
        o_c4 = general_deconv3d(o_r6, [1, 64, 64, 64, dim * 2], dim * 2, ks, ks, ks, 2, 2, 2, 0.02, "SAME", "c4")
        o_c5 = general_deconv3d(o_c4, [1, 128, 128, 128, dim], dim, ks, ks, ks, 2, 2, 2, 0.02, "SAME", "c5")
        o_c5_pad = tf.pad(o_c5, [[0, 0], [ks, ks], [ks, ks],[ks, ks], [0, 0]], "REFLECT")
        o_c6 = general_conv3d(o_c5_pad, img_layer, f, f, f, 1, 1, 1, 0.02, "VALID", "c6", do_relu=False)

        out_gen = tf.nn.tanh(o_c6, "t1")
        return out_gen


def build_generator_resnet_9blocks(inputgen, dim, name="generator"):
    with tf.variable_scope(name):
        f = 7
        ks = 3

        o_c1 = general_conv3d(inputgen, dim, f, f, f, 1, 1, 1, 0.02, "SAME", name="c1")
        o_c2 = general_conv3d(o_c1, dim * 2, ks, ks, ks, 2, 2, 2, 0.02, "SAME", "c2")
        o_c3 = general_conv3d(o_c2, dim * 4, ks, ks, ks, 2, 2, 2, 0.02, "SAME", "c3")

        o_r1 = build_resnet_block(o_c3, dim * 4, "r1")
        o_r2 = build_resnet_block(o_r1, dim * 4, "r2")
        o_r3 = build_resnet_block(o_r2, dim * 4, "r3")
        o_r4 = build_resnet_block(o_r3, dim * 4, "r4")
        o_r5 = build_resnet_block(o_r4, dim * 4, "r5")
        o_r6 = build_resnet_block(o_r5, dim * 4, "r6")
        o_r7 = build_resnet_block(o_r6, dim * 4, "r7")
        o_r8 = build_resnet_block(o_r7, dim * 4, "r8")
        o_r9 = build_resnet_block(o_r8, dim * 4, "r9")

        o_c4 = general_deconv3d(o_r9, [1, 128, 128,32, dim * 2], dim * 2, ks, ks, ks, 2, 2, 2, 0.02, "SAME", "c4")
        o_c5 = general_deconv3d(o_c4, [1, 256, 256,64, dim], dim, ks, ks, ks, 2, 2, 2, 0.02, "SAME", "c5")
        o_c6 = general_conv3d(o_c5, img_layer, f, f, f, 1, 1, 1, 0.02, "SAME", "c6", do_relu=False)
        out_gen = tf.nn.relu(o_c6, "t1")-1
        return out_gen


def build_gen_discriminator(inputdisc, dim, name="discriminator"):
    with tf.variable_scope(name):
        f = 4
        o_c1 = general_conv3d(inputdisc, dim, f, f, f, 2, 2, 2, 0.02, "SAME", "c1", do_norm=False, relufactor=0.2)
        o_c2 = general_conv3d(o_c1, dim * 2, f, f, f, 2, 2, 2, 0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = general_conv3d(o_c2, dim * 4, f, f, f, 2, 2, 2, 0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = general_conv3d(o_c3, dim * 8, f, f, f, 1, 1, 1, 0.02, "SAME", "c4", relufactor=0.2)
        o_c5 = general_conv3d(o_c4, 1, f, f, f, 1, 1, 1, 0.02, "SAME", "c5", do_norm=False, do_relu=False)

        return o_c5


