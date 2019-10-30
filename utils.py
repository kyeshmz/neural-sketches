import cStringIO
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
import cv2
import random


def truncated_z_sample(batch_size, truncation=1., seed=None):
  state = None if seed is None else np.random.RandomState(seed)
  values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state)
  return truncation * values

def one_hot(index, vocab_size=vocab_size):
  index = np.asarray(index)
  if len(index.shape) == 0:
    index = np.asarray([index])
  assert len(index.shape) == 1
  num = index.shape[0]
  output = np.zeros((num, vocab_size), dtype=np.float32)
  output[np.arange(num), index] = 1
  return output

def one_hot_if_needed(label, vocab_size=vocab_size):
  label = np.asarray(label)
  if len(label.shape) <= 1:
    label = one_hot(label, vocab_size)
  assert len(label.shape) == 2
  return label


def sample(sess, noise, label, truncation=1., batch_size=8,
           vocab_size=vocab_size):
  noise = np.asarray(noise)
  label = np.asarray(label)
  num = noise.shape[0]
  if len(label.shape) == 0:
    label = np.asarray([label] * num)
  if label.shape[0] != num:
    raise ValueError('Got # noise samples ({}) != # label samples ({})'
                     .format(noise.shape[0], label.shape[0]))
  label = one_hot_if_needed(label, vocab_size)
  ims = []
  for batch_start in tqdm(xrange(0, num, batch_size)):
    s = slice(batch_start, min(num, batch_start + batch_size))
    feed_dict = {input_z: noise[s], input_y: label[s], input_trunc: truncation}
    ims.append(sess.run(output, feed_dict=feed_dict))
  ims = np.concatenate(ims, axis=0)
  assert ims.shape[0] == num
  ims = np.clip(((ims + 1) / 2.0) * 256, 0, 255)
  ims = np.uint8(ims)
  return ims

def interpolate(A, B, num_interps):
  alphas = np.linspace(0, 1, num_interps)
  if A.shape != B.shape:
    raise ValueError('A and B must have the same shape to interpolate.')
  return np.array([(1-a)*A + a*B for a in alphas])

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

def interpolate_and_shape(A, B, num_samples, num_interps):
  interps = interpolate(A, B, num_interps)
  return (interps.transpose(1, 0, *range(2, len(interps.shape)))
                 .reshape(num_samples * num_interps, -1))

def get_interpolated_yz(categories_all, num_interps, noise_seed_A, noise_seed_B, truncation):
  nt = len(categories_all)
  num_samples = 1
  z_A, z_B = [truncated_z_sample(num_samples, truncation, noise_seed)
              for noise_seed in [noise_seed_A, noise_seed_B]]
  y_interps = []
  for i in range(nt):
    category_A, category_B = categories_all[i], categories_all[(i+1)%nt]
    y_A, y_B = [one_hot([category] * num_samples) for category in [category_A, category_B]]
    y_interp = interpolate_and_shape(np.array(y_A), np.array(y_B), num_samples, num_interps)
    y_interps.append(y_interp)

  y_interp = np.vstack(y_interps)
  z_interp = interpolate_and_shape(z_A, z_B, num_samples, num_interps * nt)
  
  return y_interp, z_interp

def get_transition_yz(classes, num_interps, truncation):
  noise_seed_A, noise_seed_B = random.uniform(10,100), random.uniform(20,120)   # fix this!
  return get_interpolated_yz(classes, num_interps, noise_seed_A, noise_seed_B, truncation=truncation)

def get_random_yz(num_classes, num_interps, truncation):
  random_classes = [ int(1000*random.random()) for i in range(num_classes) ]
  #random_classes = random.sample(insect_list,random.uniform(2,9))
  return get_transition_yz(random_classes, num_interps, truncation=truncation)

def get_combination_yz(categories, noise_seed, truncation):
  z = np.vstack([truncated_z_sample(1, truncation, noise_seed)] * (len(categories)+1))
  y = np.zeros((len(categories)+1, 1000))
  for i, c in enumerate(categories):
    y[i, c] = 1.0
    y[len(categories), c] = 1.0
  return y, z

def slerp(A, B, num_interps):  # see https://en.wikipedia.org/wiki/Slerp
  alphas = np.linspace(-1.5, 2.5, num_interps) # each unit step tends to be a 90 degree rotation in high-D space, so this is ~360 degrees
  omega = np.zeros((A.shape[0],1))
  for i in range(A.shape[0]):
      tmp = np.dot(A[i],B[i])/(np.linalg.norm(A[i])*np.linalg.norm(B[i]))
      omega[i] = np.arccos(np.clip(tmp,0.0,1.0))+1e-9
  return np.array([(np.sin((1-a)*omega)/np.sin(omega))*A + (np.sin(a*omega)/np.sin(omega))*B for a in alphas])

def slerp_and_shape(A, B, num_interps):
  interps = slerp(A, B, num_interps)
  return (interps.transpose(1, 0, *range(2, len(interps.shape)))
                 .reshape(num_interps, *interps.shape[2:]))

# def imshow(a, format='png', jpeg_fallback=True):
#   a = np.asarray(a, dtype=np.uint8)
#   str_file = cStringIO.StringIO()
#   PIL.Image.fromarray(a).save(str_file, format)
#   png_data = str_file.getvalue()
#   try:
#     disp = IPython.display.display(IPython.display.Image(png_data))
#   except IOError:
#     if jpeg_fallback and format != 'jpeg':
#       print ('Warning: image was too large to display in format "{}"; '
#              'trying jpeg instead.').format(format)
#       return imshow(a, format='jpeg')
#     else:
#       raise
#   return disp


def save_images(imgs, gdrive_folder=None):
  if gdrive_folder is not None:
    root_dir = '/content/gdrive/My Drive/%s/'%gdrive_folder
  else:
    root_dir = ''
  for i, img in enumerate(imgs):
    filename = '%sframe_%05d.png'%(root_dir, i+1)
    scipy.misc.imsave(filename, img)
    #if gdrive_folder is None:
      #files.download(filename)

def make_video(video_name, imgs):
  _, height, width, _ = imgs.shape
  video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M','J','P','G'), fps=24, frameSize=(width,height))
  for iter in range(0,imgs.shape[0]):
      video.write(imgs[iter,:,:,::-1])
  cv2.destroyAllWindows()
  video.release()
  #files.download(video_name)


def make_video_from_samples(video_name, sess, noise, label, truncation=1.0, batch_size=8, vocab_size=vocab_size):
  height, width = 512, 512
  video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M','J','P','G'), fps=30, frameSize=(width,height))
  noise = np.asarray(noise)
  label = np.asarray(label)
  num = noise.shape[0]
  if len(label.shape) == 0:
    label = np.asarray([label] * num)
  if label.shape[0] != num:
    raise ValueError('Got # noise samples ({}) != # label samples ({})'
                     .format(noise.shape[0], label.shape[0]))
  label = one_hot_if_needed(label, vocab_size)
  ims = []
  for batch_start in tqdm(xrange(0, num, batch_size)):
    s = slice(batch_start, min(num, batch_start + batch_size))
    feed_dict = {input_z: noise[s], input_y: label[s], input_trunc: truncation}
    ims = [sess.run(output, feed_dict=feed_dict)]
    ims = np.concatenate(ims, axis=0)
    ims = np.clip(((ims + 1) / 2.0) * 256, 0, 255)
    ims = np.uint8(ims)
    for iter in range(0,ims.shape[0]):
      video.write(ims[iter,:,:,::-1])
  cv2.destroyAllWindows()
  video.release()
 # files.download(video_name)

def make_images_from_samples(output_name, sess, noise, label, truncation=1.0, batch_size=8, vocab_size=vocab_size):
  height, width = 512, 512
  noise = np.asarray(noise)
  label = np.asarray(label)
  num = noise.shape[0]
  if len(label.shape) == 0:
    label = np.asarray([label] * num)
  if label.shape[0] != num:
    raise ValueError('Got # noise samples ({}) != # label samples ({})'
                     .format(noise.shape[0], label.shape[0]))
  label = one_hot_if_needed(label, vocab_size)
  ims = []
  video = None
  for batch_start in tqdm(range(0, num, batch_size)):
    s = slice(batch_start, min(num, batch_start + batch_size))
    feed_dict = {input_z: noise[s], input_y: label[s], input_trunc: truncation}
    ims = [sess.run(output, feed_dict=feed_dict)]
    ims = np.concatenate(ims, axis=0)
    ims = np.clip(((ims + 1) / 2.0) * 256, 0, 255)
    ims = np.uint8(ims)
    if video is None:
      _, height, width, _ = ims.shape
      video = cv2.VideoWriter("%s.mp4"%output_name, cv2.VideoWriter_fourcc('M','J','P','G'), fps=24, frameSize=(width,height))
      print output_name
    for iter in range(0,ims.shape[0]):
      video.write(ims[iter,:,:,::-1])
  #cv2.destroyAllWindows()
  video.release()
  #files.download("%s.mov"%output_name)


#https://twitter.com/zaidalyafeai/status/1062783610296721409
#https://colab.research.google.com/github/zaidalyafeai/Notebooks/blob/master/BigGAN.ipynb#scrollTo=h7tClPCnDCFQ

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