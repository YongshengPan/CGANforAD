import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from scipy.misc import imsave
from scipy import io as sio
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import os
import shutil
from PIL import Image
import random
import time
import sys
import csv

from layers import *
from modelcg import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_stats = 'eval'
to_restore = False
task_name = 'cyc_dis'

batch_size = 1
img_width = 144
img_height = 176
img_depth = 144
img_layer = 1

output_path = "./outputcg/samples/ADNIO/"
check_dir = "./outputcg/adni_ckpts/"
outputA_path = output_path + task_name + "/fake_A/"
outputB_path = output_path + task_name + "/fake_B/"
outputAB_path = output_path + task_name + "/smp_AB/"
input_path = "input/ADNIO/"

chkpt_fname = check_dir + task_name + '-100'
max_epoch = 101
max_images = 10000

save_training_images = True
num_of_krnl = [8, 16] % change to [16, 32]

class CycleGAN():
    def inputAB(self, imdb, cycload=True):
        flnm, grp = imdb
        if flnm in self.datapool:
            mdata, pdata = self.datapool[flnm]
        else:
            mfile = 'MRI/' + flnm + '.mat'
            pfile = 'PET/' + flnm + '.mat'

            if os.path.exists(input_path + mfile):
                mdata = np.minimum(128, np.array(sio.loadmat(input_path + mfile)['IMG'][20:164, 22:198, 9:153]))
            else:
                mdata = None
            if os.path.exists(input_path + pfile):
                pdata = np.array(sio.loadmat(input_path + pfile)['IMG'][20:164, 22:198, 9:153])
            else:
                pdata = None
            if cycload:
                self.datapool[flnm] = mdata, pdata

        if mdata is None:
            im_m = None
        else:
            im_m = np.reshape(mdata.astype(np.float32) / 64 - 1.0, (batch_size, img_width, img_height, img_depth, img_layer))
        if pdata is None:
            im_p = None
        else:
            im_p = np.reshape(pdata.astype(np.float32) / 128 - 1.0, (batch_size, img_width, img_height, img_depth, img_layer))
        return im_m, im_p


    def get_database(self, imdbname):
        imdb = []
        with open(imdbname, newline='') as csvfile:
            imdbreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in imdbreader:
                if row[1] != 'Subject':
                    imdb.append(row[1:3])
        return imdb

    def input_setup_adni(self, input_path):
        self.datapool = {}
        self.imdb_train = self.get_database(input_path + '/ADNI1_imdb_18m.csv')
        self.imdb_test  = self.get_database(input_path + '/ADNI2_imdb_18m.csv')

        print(len(self.imdb_train))
        print(len(self.imdb_test))


    def model_setup(self):
        ''' This function sets up the model to train
        self.input_A/self.input_B -> Set of training images.
        self.fake_A/self.fake_B -> Generated images by corresponding generator of input_A and input_B
        self.lr -> Learning rate variable
        self.cyc_A/ self.cyc_B -> Images generated after feeding self.fake_A/self.fake_B to corresponding generator. This is use to calcualte cyclic loss
        '''

        self.input_A = tf.placeholder(tf.float32, [None, img_width, img_height, img_depth, img_layer], name="input_A")
        self.input_B = tf.placeholder(tf.float32, [None, img_width, img_height, img_depth, img_layer], name="input_B")

        self.fake_pool_A = tf.placeholder(tf.float32, [None, img_width, img_height, img_depth, img_layer], name="fake_pool_A")
        self.fake_pool_B = tf.placeholder(tf.float32, [None, img_width, img_height, img_depth, img_layer], name="fake_pool_B")

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.num_fake_inputs = 0

        self.lr = tf.placeholder(tf.float32, shape=[], name="lr")

        with tf.variable_scope("Model") as scope:
            self.fake_B = build_generator_resnet_3blocks(self.input_A, num_of_krnl[0], name="g_A")
            self.fake_A = build_generator_resnet_3blocks(self.input_B, num_of_krnl[0], name="g_B")
            self.rec_A = build_gen_discriminator(self.input_A, num_of_krnl[1], "d_A")
            self.rec_B = build_gen_discriminator(self.input_B, num_of_krnl[1], "d_B")

            scope.reuse_variables()
            self.fake_rec_A = build_gen_discriminator(self.fake_A, num_of_krnl[1], "d_A")
            self.fake_rec_B = build_gen_discriminator(self.fake_B, num_of_krnl[1], "d_B")
            self.cyc_A = build_generator_resnet_3blocks(self.fake_B, num_of_krnl[0], "g_B")
            self.cyc_B = build_generator_resnet_3blocks(self.fake_A, num_of_krnl[0], "g_A")

            # scope.reuse_variables()
            self.fake_pool_rec_A = build_gen_discriminator(self.fake_pool_A, num_of_krnl[1], "d_A")
            self.fake_pool_rec_B = build_gen_discriminator(self.fake_pool_B, num_of_krnl[1], "d_B")

    def loss_calc(self):

        ''' In this function we are defining the variables for loss calcultions and traning model

        d_loss_A/d_loss_B -> loss for discriminator A/B
        g_loss_A/g_loss_B -> loss for generator A/B
        *_trainer -> Variaous trainer for above loss functions
        *_summ -> Summary variables for above loss functions'''

        cyc_loss_A = tf.reduce_mean(tf.abs(self.input_A - self.cyc_A))
        cyc_loss_B = tf.reduce_mean(tf.abs(self.input_B - self.cyc_B))
        disc_loss_A = tf.reduce_mean(tf.abs(self.fake_rec_A-1))
        disc_loss_B = tf.reduce_mean(tf.abs(self.fake_rec_B-1))

        if task_name == 'dis':
            g_loss_A = disc_loss_B
            g_loss_B = disc_loss_A
        elif task_name == 'cyc':
            g_loss_A = cyc_loss_A
            g_loss_B = cyc_loss_B
        elif task_name == 'cyc_dis':
            g_loss_A = cyc_loss_A + disc_loss_B
            g_loss_B = cyc_loss_B + disc_loss_A
        else:
            print('unknown')
            g_loss_A = cyc_loss_A + disc_loss_B
            g_loss_B = cyc_loss_B + disc_loss_A

        d_loss_A = (tf.reduce_mean(tf.square(self.fake_pool_rec_A)) + tf.reduce_mean(
            tf.squared_difference(self.rec_A, 1))) / 2.0
        d_loss_B = (tf.reduce_mean(tf.square(self.fake_pool_rec_B)) + tf.reduce_mean(
            tf.squared_difference(self.rec_B, 1))) / 2.0

        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)

        self.model_vars = tf.trainable_variables()

        d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
        g_A_vars = [var for var in self.model_vars if 'g_A' in var.name]
        d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
        g_B_vars = [var for var in self.model_vars if 'g_B' in var.name]

        self.d_A_trainer = optimizer.minimize(d_loss_A, var_list=d_A_vars)
        self.d_B_trainer = optimizer.minimize(d_loss_B, var_list=d_B_vars)
        self.g_A_trainer = optimizer.minimize(g_loss_A, var_list=g_A_vars)
        self.g_B_trainer = optimizer.minimize(g_loss_B, var_list=g_B_vars)

        for var in self.model_vars: print(var.name)

        # Summary variables for tensorboard

        self.g_A_loss_summ = tf.summary.scalar("g_A_loss", g_loss_A)
        self.g_B_loss_summ = tf.summary.scalar("g_B_loss", g_loss_B)
        self.d_A_loss_summ = tf.summary.scalar("d_A_loss", d_loss_A)
        self.d_B_loss_summ = tf.summary.scalar("d_B_loss", d_loss_B)

    def save_training_images(self, sess, epoch):

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for ptr in range(0, max_images):
            inputA,inputB = self.inputAB(self.imdb_train[ptr], cycload=True)
            if (inputA is not None) & (inputB is not None):
                fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run(
                        [self.fake_A, self.fake_B, self.cyc_A, self.cyc_B],
                        feed_dict={self.input_A: inputA[0:1,:,:,:], self.input_B: inputB[0:1,:,:,:]})
                sio.savemat(output_path+"/fake_" + str(epoch) + "_" + str(ptr) + ".mat",
                            {'fake_A': fake_A_temp[0], 'fake_B': fake_B_temp[0],
                             'cyc_A': cyc_A_temp[0], 'cyc_B': cyc_B_temp[0],
                             'input_A': inputA[0], 'input_B': inputB[0]})
                break

    def train(self):
        ''' Training Function '''

        self.input_setup_adni(input_path)  # Load Dataset from the dataset folder
        self.model_setup()  # Build the network
        self.loss_calc()    # Loss function calculations

        # Initializing the global variables
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        saver = tf.train.Saver(max_to_keep=0)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init)
            # Restore the model to run the model from last checkpoint
            if to_restore:
                # chkpt_fname = tf.train.latest_checkpoint(check_dir)
                saver.restore(sess, chkpt_fname)
            writer = tf.summary.FileWriter("./outputcg/event")

            if not os.path.exists(check_dir):
                os.makedirs(check_dir)

            # Training Loop
            for epoch in range(sess.run(self.global_step), max_epoch):
                print("In the epoch ", epoch)
                saver.save(sess, os.path.join(check_dir, task_name), global_step=epoch)
                # Dealing with the learning rate as per the epoch number
                curr_lr = 0.0002*(200-max(epoch, 100))/100

                if (save_training_images):
                    self.save_training_images(sess, epoch)

                for ptr in range(0, min(max_images, len(self.imdb_train))):
                    print("In the iteration ", ptr)

                    print(self.imdb_train[ptr])
                    inputA, inputB = self.inputAB(self.imdb_train[ptr],  cycload=True)
                    if inputA is None: continue
                    if inputB is None: continue
                    for idx in range(inputA.shape[0]):
                        # Optimizing the G_A network
                        _, fake_B_temp, summary_str = sess.run([self.g_A_trainer, self.fake_B, self.g_A_loss_summ],
                                                           feed_dict={self.input_A: inputA[idx:idx+1], self.input_B: inputB[idx:idx+1],
                                                                      self.lr: curr_lr})
                        writer.add_summary(summary_str, epoch * max_images + ptr)

                        if 'dis' in task_name:
                            # Optimizing the D_B network
                            _, summary_str = sess.run([self.d_B_trainer, self.d_B_loss_summ],
                                              feed_dict={self.input_A: inputA[idx:idx+1], self.input_B: inputB[idx:idx+1], self.lr: curr_lr,
                                                         self.fake_pool_B: fake_B_temp})
                            writer.add_summary(summary_str, epoch * max_images + ptr)

                        # Optimizing the G_B network
                        _, fake_A_temp, summary_str = sess.run([self.g_B_trainer, self.fake_A, self.g_B_loss_summ],
                                                           feed_dict={self.input_A: inputA[idx:idx+1], self.input_B: inputB[idx:idx+1],
                                                                      self.lr: curr_lr})
                        writer.add_summary(summary_str, epoch * max_images + ptr)

                        if 'dis' in task_name:
                            # Optimizing the D_A network
                            _, summary_str = sess.run([self.d_A_trainer, self.d_A_loss_summ],
                                              feed_dict={self.input_A: inputA[idx:idx+1], self.input_B: inputB[idx:idx+1], self.lr: curr_lr,
                                                         self.fake_pool_A: fake_A_temp})
                            writer.add_summary(summary_str, epoch * max_images + ptr)

                        self.num_fake_inputs += 1
                sess.run(tf.assign(self.global_step, epoch + 1))
            writer.add_graph(sess.graph)

    def test(self):

        ''' Testing Function'''

        print("Testing the results")
        self.input_setup_adni(input_path)
        self.model_setup()
        saver = tf.train.Saver()
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.6
        config.gpu_options.allow_growth = True

        with tf.Session() as sess:
            sess.run(init)
            print(chkpt_fname)
            saver.restore(sess, chkpt_fname)
            if not os.path.exists(outputA_path):
                os.makedirs(outputA_path)
            if not os.path.exists(outputB_path):
                os.makedirs(outputB_path)
            img_out = np.zeros([181, 217, 181], np.uint8)
            for ptr in range(min(len(self.imdb_test), max_images)):
                inputA, inputB = self.inputAB(self.imdb_test[ptr], cycload=False)
                if inputA is None: continue
                filename = self.imdb_test[ptr][0]
                fake_B_temp = sess.run(self.fake_B, feed_dict={self.input_A: inputA})
                img_out[20:164, 20:200, 9:153] = ((np.squeeze(fake_B_temp) + 1) * 128).astype(np.uint8)
                sio.savemat(outputA_path + filename + '.mat', {'IMG': img_out})
                print(filename)

            for ptr in range(min(len(self.imdb_test), max_images)):
                inputA, inputB = self.inputAB(self.imdb_test[ptr], cycload=False)
                if inputB is None: continue
                filename = self.imdb_test[ptr][0]
                fake_A_temp = sess.run(self.fake_A, feed_dict={self.input_B: inputB})
                img_out[20:164, 20:200, 9:153] = ((np.squeeze(fake_A_temp) + 1) * 64).astype(np.uint8)
                sio.savemat(outputB_path + filename + '.mat',
                            {'IMG': img_out})
                print(filename)

    def eval(self):
        ''' Testing Function'''
        print("Testing the results")
        self.input_setup_adni(input_path)
        self.model_setup()
        saver = tf.train.Saver()
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session() as sess:
            sess.run(init)
            print(chkpt_fname)
            saver.restore(sess, chkpt_fname)
            if not os.path.exists(outputAB_path):
                os.makedirs(outputAB_path)
            MAE = []; SSIM = []; PSNR = []
            for ptr in range(min(len(self.imdb_test), max_images)):
                inputA, inputB = self.inputAB(self.imdb_test[ptr], cycload=False)
                if (inputB is None) | (inputA is None): continue
                filename = self.imdb_test[ptr][1]+self.imdb_test[ptr][0]
                print(filename)
                fake_A, fake_B = sess.run([self.fake_A, self.fake_B], feed_dict={self.input_A: inputA[0:1], self.input_B: inputB[0:1]})
                print([np.mean(np.abs(fake_A-inputA)), np.mean(np.abs(fake_B-inputB))])
                MAE.append([np.mean(np.abs(fake_A-inputA)), np.mean(np.abs(fake_B-inputB))])
                SSIM.append([ssim(inputA[0], fake_A[0], multichannel=True), ssim(inputB[0], fake_B[0], multichannel=True)])
                PSNR.append([psnr(inputA[0]/2, fake_A[0]/2), psnr(inputB[0]/2, fake_B[0]/2)])
                imsave(os.path.join(outputAB_path, filename + '.bmp'), np.concatenate(
                    (np.array((inputA[:, :, :, 72, :]) * 100 + 100).astype(np.uint8).reshape([img_width, img_height]),
                    np.array((inputB[:, :, :, 72, :]) * 100 + 100).astype(np.uint8).reshape([img_width, img_height]),
                    np.array((fake_A[:, :, :, 72, :]) * 100 + 100).astype(np.uint8).reshape([img_width, img_height]),
                    np.array((fake_B[:, :, :, 72, :]) * 100 + 100).astype(np.uint8).reshape([img_width, img_height])), axis=1), 'bmp')
            print(np.mean(MAE, axis=0), np.mean(SSIM, axis=0), np.mean(PSNR, axis=0))
            print(np.std(MAE, axis=0), np.std(SSIM, axis=0), np.std(PSNR, axis=0))


def main():
    model = CycleGAN()
    if model_stats == 'train':
        model.train()
    elif model_stats == 'test':
        model.test()
    else:
        model.eval()


if __name__ == '__main__':
    main()