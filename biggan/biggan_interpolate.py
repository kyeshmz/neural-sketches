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
import cv2
import random


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


# In[4]:


vocab_size = inputs['y'].get_shape().as_list()[1]
latent_size = inputs['z'].get_shape().as_list()[1]

print('Number of labels ', vocab_size)
print('The size of the latent space ', latent_size)


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


# In[6]:


one_hot(10)



# In[7]:


def get_zy(index, trunc = 1., batch_size = 1, seed = None):
    #convert the label to one hot encoding 
    y = one_hot(index)

    #sample a batch of z-vectors 
    z = truncnorm.rvs(-2, 2, size=(batch_size, latent_size), random_state = np.random.RandomState(seed)) * trunc
    return z, y


# In[8]:


z, _ = get_zy(np.random.choice(range(0, vocab_size), size = 1),  seed = 2)

y = np.linspace(0, latent_size, latent_size)

plt.scatter(y, z)
plt.show()


# In[17]:


# def postprocess(img):
#     img = np.clip(((img + 1) / 2.0) * 256, 0, 255)
#     img = np.uint8(img)  
#     img = img.squeeze()
#     return img
# def generate(sess, z, y, trunc = 1.):
#     feed_dict = {inputs['z']: z, inputs['y']:y, inputs['truncation']: trunc}
#     im = sess.run(output, feed_dict=feed_dict)

#     #postprocess the image 
# #     im = np.clip(((im + 1) / 2.0) * 256, 0, 255)
# #     im = np.uint8(im)  
# #     im = im.squeeze()
#     im = postprocess(im)
#     return im


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

# In[27]:


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


# In[30]:

animals = [ f for f in range(0,397)]
n_labels = 1000 #BigGan is trained on 1000 lables const
z_dim = 128 # the z vector dimension  const
num_interps =  450# the number of interpolations between each two consectuve images, basically FPS x seconds 14 83
num_interps_images = 4#the number of images, or number of images to be ping ponged
r = 71 #the class to interpolate images from  


birdcat = [f for f in range(82,100)]
otherbirds = [g for g in range(127,146)]
for bird in otherbirds:
    birdcat.append(bird)

# In[42]:
   

def create_animation(images):
    imageio.mimsave('./animation.gif', images)
    with open('./animation.gif','rb') as f:
        display.display(display.Image(data=f.read(), height=512))


def make_video(video_name, imgs):
    _, height, width, _ = imgs.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M','J','P','G'), fps=24, frameSize=(width,height))
    for iter in range(0,imgs.shape[0]):
        video.write(imgs[iter,:,:,::-1])
    cv2.destroyAllWindows()
    video.release()

def make_video_list(video_name, imgs):
    for img in imgs:
        print("img", img)
        print("img type", type(img))
        _, height, width,= img.shape
        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M','J','P','G'), fps=24, frameSize=(width,height))
        for iter in range(0,img.shape[0]):
            video.write(imgs[iter,:,:,::-1])
        cv2.destroyAllWindows()
        video.release()

# In[44]:
import os
import cv2
from multiprocessing import Pool
allimages= []

def writegif(img):
    (filename, image) = img
    for frame in image:
        imageio.imwrite(filename, frame, duration=14)

def writeimage(img):
    (filename, image) = img
    imageio.imwrite(filename, image)

def writevideo(img):
    (filename, images) = img
    for image in images:
        video.write(cv2.imread(os.path.join(file, image)))

        theframe = cv2.imread(frame)
        height, width, layers = theframe.shape
        video = cv2.VideoWriter(video_name, 0, 1, (width,height))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for f in range(0,200):
        print(f)
        #r = np.random.randint(low=0,high=397)
        r = random.choice(birdcat)
        images = []

        #initial interpolation
        zA, yA = get_zy(r)
        izA, iyA = zA, yA 

        for i in range(0, num_interps_images):

            #use the first interpolation once we reach the last image to create a loop effect
            if i == num_interps_images - 1:
                zB, yB = izA, iyA
            else:
                zB, yB = get_zy(r)

            #create interpolation for both the category and the z vector 
            interps_z = interpolate_hypersphere(zA, zB, num_interps)
            interps_y = interpolate_hypersphere(yA, yB, num_interps)

            #create an image for each interpolation 
            for i in range(0, len(interps_z)):
                im = generate(sess, interps_z[i], interps_y[i], trunc = 0.2, squeeze=True) 
                
                # feed_dict = {inputs['z']: interps_z[i], inputs['y']:interps_y[i], inputs['truncation']: 0.3}
                # im = sess.run(output, feed_dict=feed_dict)
                # im = np.clip(((im + 1) / 2.0) * 256, 0, 255)
                # im = np.uint8(im)
                # print("im",im)
                # print("im type", type(im))
                
                images.append(im)

                #print(type(im))
                filename = '%d_%d_animation_%d.mp4' %(f,i,r)

                filedir = os.path.join('./hatra',filename)
                #writer = imageio.get_writer('test.mp4', fps=30)
                #writer.append_data(imageio.imread(im.squeeze()))
                # writer.close()


                

            #save the last interpolated vector to use for the next image
            zA = interps_z[-1]
            yA = interps_y[-1] 
        #its images[1:] because its the first image is from the last image
        allimages.append((filedir,images[1:]))
        #imageio.mimsave(filename, images[1:], fps=60)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(filename, fourcc, 30, (512,512))
        for image in images[1:]:
            # bridging PIL to opencv for video
            imagetemp = image.copy()
            video.write(cv2.cvtColor(np.array(imagetemp), cv2.COLOR_RGB2BGR))
        cv2.destroyAllWindows()
        video.release()


        #imageio.mimsave(filedir, images[1:])
        # writer = imageio.get_writer('test.mp4', fps=30)
        # # for f in images[1:]:
        # #     writer.append_data(imageio.imread(f))
        # writer.append_data(images[1:])
        # writer.close()
        #writer.append_data(images[1:])

#        allimages.append(images[1:])
#        with open(filename,'rb') as f:
#            display.display(display.Image(data=f.read(), height=512))
# import multiprocessing
# p = multiprocessing.Pool(44)
# flat_list = [item for sublist in allimages for item in sublist]
# result = p.map(writegif, flat_list)

# In[ ]:




