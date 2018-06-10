import os
import sys
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tqdm import tqdm

from utils import load_data
from utils import save_images
from utils import shuffle

from capsLayer import CapsLayer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

epoch_size = 100000
batch_size = 128
data_shape = (28, 28)
channel = 1

dataset_name = 'mnist'

X = tf.placeholder(tf.float32, shape=(batch_size, data_shape[0], data_shape[1], channel))
label = tf.placeholder(tf.uint8, shape=(batch_size, ))
Y = tf.one_hot(label, depth=10, axis=1, dtype=tf.float32)

# arch
# X = [batch_size, 28, 28, 1]
conv1 = slim.conv2d(X, 256, [9, 9], scope='Conv1_1', padding='VALID')
# conv1 = [batch_size, 20, 20, 256]
caps1 = CapsLayer(layer_type='PrimaryCaps')
capsLayer1 = caps1(conv1, num_outputs=32, vec_len=8, kernel_size=9, stride=2)
# capsLayer1 = [batch_size, 6 * 6 * 32, 8, 1]
caps2 = CapsLayer(layer_type='DigitCaps')
capsLayer2 = caps2(capsLayer1, num_outputs=10, vec_len=16)
# capsLayer2 =[batch_size, 10, 16, 1]

# Decoder
v_length = tf.sqrt(tf.reduce_sum(tf.square(capsLayer2), axis=2, keepdims=True) + 1e-9)
softmax_v = tf.nn.softmax(v_length, axis=1)
# softmax_v = [batch_size, 10, 1, 1]
argmax_idx = tf.reshape(tf.to_int32(tf.argmax(softmax_v, axis=1)), shape=(batch_size, ))
print(argmax_idx)
# argmax_idx = [batch_size, 1]

masked_v = []
for batch in range(batch_size):
    v = capsLayer2[batch][argmax_idx[batch], :]
    # v = [16, 1]
    masked_v.append(tf.reshape(v, shape=(1, 1, 16, 1)))

masked_vec = tf.concat(masked_v, axis=0)
# masked_vec = [batch_size, 1, 16, 1]
"""
# mask true label
masked_vec = tf.multiply(tf.squeeze(capsLayer2), tf.reshape(Y, (-1, 10, 1)))
"""

vector_j = tf.reshape(masked_vec, shape=(batch_size, -1))
with tf.variable_scope('Decoder'):
    FC1 = slim.fully_connected(vector_j, 512, scope='fc1', activation_fn=tf.nn.relu)
    FC2 = slim.fully_connected(FC1, 1024, scope='fc2', activation_fn=tf.nn.relu)
    decoded = slim.fully_connected(FC2, data_shape[0] * data_shape[1], activation_fn=tf.sigmoid)

# loss

# margin loss = Tc max(0, m+ - ||vc||)^2 + lambda * (1-Tc)*max(0, ||vc - m-) ^ 2
m_plus = 0.9
m_minus = 0.1
lam = 0.5
regularization_scale = 0.392

max_l = tf.square(tf.maximum(0.0, m_plus-v_length))
max_r = tf.square(tf.maximum(0.0, v_length-m_minus))
max_l = tf.reshape(max_l, shape=(batch_size, -1))
max_r = tf.reshape(max_r, shape=(batch_size, -1))
margin_loss = tf.reduce_mean(tf.reduce_sum(Y * max_l + lam * (1 - Y) * max_r, axis=1))

# reconstruction loss
rec_loss = tf.reduce_mean(tf.square(decoded - tf.reshape(X, shape=(batch_size, -1))))

total_loss = margin_loss + regularization_scale * rec_loss

train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(total_loss)
init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.to_int32(label), argmax_idx)
train_accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32)) / batch_size

saver = tf.train.Saver(max_to_keep=20)

def train(model_dir, sample_dir):

    model_name = 'model_'

    input_x, input_y, validation_x, validation_y = load_data(dataset_name, batch_size)

    with tf.Session(config=tf.ConfigProto()) as sess:

        sess.run(init)

        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print("[*] restore model : %s"%ckpt_name)
            saver.restore(sess, os.path.join(model_dir, ckpt_name))
        else:
            print("model does not exits")

        print('============ Start training ==============')

        iteration_number = input_x.shape[0] // batch_size

        for epoch in range(epoch_size):

            start_position = int(epoch % iteration_number)

            if (start_position == 0):
                input_x, input_y = shuffle(input_x, input_y)

            train_X = input_x[start_position*batch_size : (start_position+1)*batch_size]
            train_Y = input_y[start_position*batch_size : (start_position+1)*batch_size]

            if (epoch % 10 == 0):
                _, loss, train_acc = sess.run([train_op, total_loss, train_accuracy], feed_dict={X:train_X, label:train_Y})

                print('Train [%d / %d]: accuracy %.4f, loss %.4f' % (epoch, epoch_size, train_acc, loss))
            else:
                sess.run(train_op, feed_dict={X:train_X, label:train_Y})

            # save model and sample result
            if (epoch % 20 == 0):
                
                # evaluate
                iter_num = validation_x[1:1000].shape[0] // batch_size
                acc = 0.0
                for itr in range(iter_num):
                    samples = sess.run([decoded], feed_dict={X:train_X, label:train_Y})
                    # pos = int(random.random() * 5)
                    pos = itr
                    validate_label = sess.run(argmax_idx, feed_dict={X:validation_x[pos*batch_size:(pos+1)*batch_size]})

                    validate_acc = 1.0 * np.sum(validate_label == validation_y[pos*batch_size:(pos+1)*batch_size]) / batch_size
                    acc += validate_acc

                acc = acc / iter_num
                save_images(np.reshape(samples, (batch_size, 28, 28))[0:100], [10, 10], os.path.join(sample_dir, 'sample_'+str(epoch)+'.png'))
                save_images(np.reshape(train_X, (batch_size, 28, 28))[0:100], [10, 10], os.path.join(sample_dir, 'input_'+str(epoch)+'.png'))
                saver.save(sess, './model/' + model_name + str(epoch) + '.ckpt')
                print("Save model and sample images, val acc: %.4f" % acc)
            

def test(model_dir, out_dir):
    with tf.Session() as sess:

        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print("[*] restore model : %s"%ckpt_name)
            saver.restore(sess, os.path.join(model_dir, ckpt_name))
        else:
            print("model does not exits")
            finish()

        test_X, test_Y = load_data(dataset_name, batch_size, is_training=False)

        iteration_number = test_X.shape[0] // batch_size
        print(iteration_number)

        correct_num = 0
        total_num = iteration_number * batch_size

        for itr in range(iteration_number):
            predict_label, samples = sess.run([argmax_idx, decoded], feed_dict={X:test_X[itr*batch_size:(itr+1)*batch_size]})
            correct_num += np.sum(predict_label == test_Y[itr*batch_size:(itr+1)*batch_size])
            if (itr == 0):
                save_images(np.reshape(samples[0:100], (100, 28, 28)), [10, 10], os.path.join(out_dir, 'test.png'))
                save_images(np.reshape(test_X[0:100], (100, 28, 28)), [10, 10], os.path.join(out_dir, 'input.png'))

        accuracy = 1.0 * correct_num / total_num

        print('accuracy: %.6f' % accuracy)


def main(_):

    if (sys.argv[1] == 'train'):
        model_dir = './model'
        
        sample_dir = './sample'
 
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
 
        train(model_dir, sample_dir)
    else:
        model_dir = './model'
        test_dir = './test'
        if not os.path.exists(model_dir):
            print('model does not exist')
            finish()
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        test(model_dir, test_dir)

if __name__ == "__main__":
    tf.app.run()
