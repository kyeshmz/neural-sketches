from IPython import display
import numpy as np
from scipy.stats import truncnorm
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from numpy import linalg as LA
import imageio
import cv2


def one_hot(index, vocab_size=1000):
    index = np.asarray(index)
    if len(index.shape) == 0:
        index = np.asarray([index])
    assert len(index.shape) == 1
    num = index.shape[0]
    output = np.zeros((num, vocab_size), dtype=np.float32)
    output[np.arange(num), index] = 1
    return output
   
def postprocess(img):
    img = np.clip(((img + 1) / 2.0) * 256, 0, 255)
    img = np.uint8(img)  
    img = img.squeeze()
    return img

def generate(sess, z, y, trunc = 1.):
    feed_dict = {inputs['z']: z, inputs['y']:y, inputs['truncation']: trunc}
    im = sess.run(output, feed_dict=feed_dict)

    #postprocess the image 
#     im = np.clip(((im + 1) / 2.0) * 256, 0, 255)
#     im = np.uint8(im)  
#     im = im.squeeze()
    im = postprocess(im)
    return im

def interpolate_hypersphere(v1, v2, num_steps):
    v1_norm = LA.norm(v1)
    v2_norm = LA.norm(v2)
    v2_normalized = v2 * (v1_norm / v2_norm)

    vectors = []
    for step in range(num_steps):
        interpolated = v1 + (v2_normalized - v1) * step / (num_steps - 1)
        interpolated_norm =  LA.norm(interpolated)
        interpolated_normalized = interpolated * (v1_norm / interpolated_norm)
        vectors.append(interpolated_normalized)
    return np.array(vectors)

def get_zy(index, seed = None):
    yA = one_hot(index)
    zA = truncnorm.rvs(-2, 2, size=(1, z_dim), random_state = np.random.RandomState(seed))
    return zA, yA

#helper functions to generate the encoding
def one_hot(index, vocab_size=1000):
    index = np.asarray(index)
    if len(index.shape) == 0:
        index = np.asarray([index])
    assert len(index.shape) == 1
    num = index.shape[0]
    output = np.zeros((num, vocab_size), dtype=np.float32)
    output[np.arange(num), index] = 1
    return output

def create_animation(images):
    imageio.mimsave('./animation.gif', images)
    with open('./animation.gif','rb') as f:
        display.display(display.Image(data=f.read(), height=512))