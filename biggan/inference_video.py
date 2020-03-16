#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython import display
import numpy as np
from scipy.stats import truncnorm
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from numpy import linalg as LA
import imageio
import os


# In[2]:


tf.reset_default_graph()
model_size = "3)biggan-512" #@param ["1)biggan-128" , "2)biggan-256" , "3)biggan-512"]
which_model = model_size.split(')')[1]
module_path = 'https://tfhub.dev/deepmind/'+which_model+'/2'
module = hub.Module(module_path)


# In[3]:


inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
          for k, v in module.get_input_info_dict().items()}
output = module(inputs)

print ('Inputs:\n', '\n'.join(
        '{}: {}'.format(*kv) for kv in inputs.items()))

print ('Output:', output)


# Store the number of the labels and the size of the latent space $z$ to sample from 

# In[4]:


vocab_size = inputs['y'].get_shape().as_list()[1]
latent_size = inputs['z'].get_shape().as_list()[1]

print('Number of labels ', vocab_size)
print('The size of the latent space ', latent_size)


# Note that the model takes label inputs as one hot encoded which maps each class 𝑐∈[0,1000) to a vector of size 1000 with all zeros except the index of the corrosponding class
# 

# In[5]:


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


# Truncation Trick
# 
# Previous work on GANs samples the latent vector 𝑧∼(0,𝐼) as a normal distirubtion with the identity convariance matrix. OTOH, the authors of BigGans used a truncated normal distriubtion in a certain region [−𝑎,𝑎] for 𝑎∈ℝ+where the sampled values outside that region are resampled to fall in the region. This resulted in better results of both IS and FID scores. The drawback of this is a reduction in the overall variety of vector sampling. Hence there is a trade-off between sample quality and variety ofr a given network G.
# 
# Here we set the default truncation threshold to be 2 i.e 𝑧 values are sampled from the region [−2,2]. Note that the optional variable seed takes integer values and it's used to reserve the state of randomness for resampling. Hence, if you use the same seed you will get the same 𝑧 vector .

# In[6]:


def get_zy(index, trunc = 1., batch_size = 1, seed = None):
    #convert the label to one hot encoding 
    y = one_hot(index)

    #sample a batch of z-vectors 
    z = truncnorm.rvs(-2, 2, size=(batch_size, latent_size), random_state = np.random.RandomState(seed)) * trunc
    return z, y
def truncated_z_sample(batch_size, truncation=1., seed=None):
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, latent_size), random_state=state)
    return truncation * values


# In[15]:


def postprocess(img, squeeze=True):
    img = np.clip(((img + 1) / 2.0) * 256, 0, 255)
    img = np.uint8(img)  
    if squeeze:
        img = img.squeeze()
        return img
    else:
        return img
    
def generate(sess, z, y, trunc = 1., squeeze=True):
    feed_dict = {inputs['z']: z, inputs['y']:y, inputs['truncation']: trunc}
    im = sess.run(output, feed_dict=feed_dict)

    #postprocess the image 
    im = postprocess(im, squeeze=squeeze)
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


# In[16]:


# new part
from skimage import io, data, transform
import cv2
from pathlib import Path

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

def classify_image(classifier, img):
    h, w = hub.get_expected_image_size(classifier)
    x = tf.placeholder(tf.float32, shape=(None, h, w, 3))
    y = tf.nn.softmax(classifier(x))
    data = transform.resize(img, [h, w])
    with tf.Session().as_default() as sess:
        tf.global_variables_initializer().run()
        y_pred = sess.run(y, feed_dict={x: [data]})
        return y_pred
classifier = hub.Module("https://tfhub.dev/google/imagenet/nasnet_large/classification/1")


# In[17]:


import IPython.display
import numpy as np
import urllib
import PIL.Image
from scipy.stats import truncnorm
from skimage import io, data, transform
import requests
import tensorflow as tf
import tensorflow_hub as hub
import scipy.misc
from tqdm import tqdm
from random import random

def imgrid(imarray, cols=5, pad=1):
    if imarray.dtype != np.uint8:
        raise ValueError('imgrid input imarray must be uint8')
    pad = int(pad)
    assert pad >= 0
    cols = int(cols)
    assert cols >= 1
    N, H, W, C = imarray.shape
    rows = int(np.ceil(N / float(cols)))
    batch_pad = rows * cols - N
    assert batch_pad >= 0
    post_pad = [batch_pad, pad, pad, 0]
    pad_arg = [[0, p] for p in post_pad]
    imarray = np.pad(imarray, pad_arg, 'constant', constant_values=255)
    H += pad
    W += pad
    grid = (imarray
          .reshape(rows, cols, H, W, C)
          .transpose(0, 2, 1, 3, 4)
          .reshape(rows*H, cols*W, C))
    if pad:
        grid = grid[:-pad, :-pad]
    return grid


# In[18]:


import io
def imshow(a, format='png', jpeg_fallback=True):
    a = np.asarray(a, dtype=np.uint8)
    str_file = io.BytesIO()
    PIL.Image.fromarray(a).save(str_file, format)
    png_data = str_file.getvalue()
    try:
        disp = IPython.display.display(IPython.display.Image(png_data))
    except IOError:
        if jpeg_fallback and format != 'jpeg':
            print ('Warning: image was too large to display in format "{}"; '
             'trying jpeg instead.').format(format)
            return imshow(a, format='jpeg')
        else:
            raise
            return disp


# In[19]:


images = []
datapath = os.getcwd()  + '/planet/'
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    #tf.global_variables_initializer().run()
    for i,img in enumerate(os.listdir(datapath)):
        print(img)
        print(type(img))
        
        
        
        relative = Path(os.path.join(datapath,img))
        
        filename, ext = os.path.splitext(str(img))
        
        absolute = os.path.abspath(relative)
        print(absolute)
        img = cv2.imread(absolute)
        #print(img)

        y_pred = classify_image(classifier, img)
        num_samples = 1
        truncation = 0.8
        noise_seed = 112

        z = truncated_z_sample(num_samples, truncation, noise_seed)
        y = np.vstack([y_pred[0][1:]]*num_samples)
        
        im0 = cv2.resize(img,(512, 512))
        #im1 = sample(sess, z, y, truncation=truncation)[0]
        im1 = generate(sess, z, y, trunc=truncation,squeeze=False)[0]
        ims = imgrid(np.array([im0, im1]), cols=2)
        #plt.show(ims)
        #images.append(ims)
        #imshow(ims)
        filename = '%d_%s.png' %(i,filename)
        filepath = os.path.join(os.getcwd(),'./out',filename)
        imageio.imwrite(filepath, ims)


# In[ ]:





# In[ ]:




