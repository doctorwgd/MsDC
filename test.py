import time
import os
import argparse
from glob import glob
import tensorflow as tf

from utils import *


def MsDC(input, is_training=True, output_channels=1):
    with tf.variable_scope('block1'):
        output_1 = tf.layers.conv2d(input, 64, 3, padding='same', activation=tf.nn.relu)

    for layers in range(2, 7 + 1):
        with tf.variable_scope('block%d' % layers):
            if layers == 2:
                input_tensor = output_1
            else:
                input_tensor = tf.concat((input_tensor, output),3)
            output = block_dilated(input_tensor,layers, is_training)
            print(layers)
    with tf.variable_scope('block8'):
        output = tf.layers.conv2d(output, output_channels, 3, padding='same')
    return input - output

def block_dilated(input,layers,training):
    output1 = tf.layers.conv2d(input, 32, 3, padding='same', name='conv%d' % layers, use_bias=False)
    output2 = tf.layers.conv2d(input, 32, 3, padding='same', dilation_rate = 2, name='dilated2_conv%d' % layers, use_bias=False)
    output3 = tf.layers.conv2d(input, 32, 3, padding='same', dilation_rate = 3, name='dilated3_conv%d' % layers, use_bias=False)
    logits =  tf.concat((output1,output2),3)
    logits = tf.nn.relu(tf.layers.batch_normalization(logits, training=training))
    return logits

class denoiser(object):
    def __init__(self, sess, input_c_dim=1, sigma=25, batch_size=128):
        self.sess = sess
        self.input_c_dim = input_c_dim
        self.sigma = sigma
        # build model
        self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim],
                                 name='clean_image')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.X = self.Y_ + tf.truncated_normal(shape=tf.shape(self.Y_), stddev=self.sigma / 255.0)  # noisy images
        self.Y = MsDC(self.X, is_training=self.is_training)
        self.loss = (1.0 / batch_size) * tf.nn.l2_loss(self.Y_ - self.Y)
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.eva_psnr = tf_psnr(self.Y, self.Y_)
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")

    def denoise(self, data):
        output_clean_image, noisy_image, psnr = self.sess.run([self.Y, self.X, self.eva_psnr],
                                                              feed_dict={self.Y_: data, self.is_training: False})
        return output_clean_image, noisy_image, psnr

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0

    def test(self, test_files, ckpt_dir, save_dir):
        """Test MsDC"""
        # init variables
        tf.initialize_all_variables().run()
        assert len(test_files) != 0, 'No testing data!'
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        psnr_sum = 0
        time_sum = 0
        print("[*] " + 'noise level: ' + str(self.sigma) + " start testing...")
        
        for idx in range(len(test_files)):
            clean_image = load_images(test_files[idx]).astype(np.float32) / 255.0
            start_time = time.time()
            output_clean_image, noisy_image = self.sess.run([self.Y, self.X],
                                                            feed_dict={self.Y_: clean_image, self.is_training: False})
            eve_time  = time.time() - start_time
            if idx!=0:
               time_sum +=eve_time 
            groundtruth = np.clip(255 * clean_image, 0, 255).astype('uint8')
            noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            print("img%d PSNR: %.2f" % (idx, psnr))
            psnr_sum += psnr
            save_images(os.path.join(save_dir, 'noisy%d.png' % idx), noisyimage)
            save_images(os.path.join(save_dir, 'denoised%d.png' % idx), outputimage)
        avg_psnr = psnr_sum / len(test_files)
        avg_time = (time_sum) / (len(test_files)-1)
        print("--- Average PSNR %.2f ---" % avg_psnr)
        print("--- test time: %4.4f" % avg_time)



parser = argparse.ArgumentParser(description='')

parser.add_argument('--sigma', dest='sigma', type=int, default=25, help='noise level')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint_demo', help='models are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--test_set', dest='test_set', default='Set12', help='dataset for testing')
args = parser.parse_args()




def denoiser_test(denoiser):
    test_files = glob('./data/test/Set12/*.png')
    denoiser.test(test_files, ckpt_dir=args.ckpt_dir, save_dir=args.test_dir)


def main(_):

    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = denoiser(sess, sigma=args.sigma)
        denoiser_test(model)


if __name__ == '__main__':
    tf.app.run()