!pip install -U git+https://github.com/bmcfee/RasterFairy/ --user

%matplotlib inline
import os
import random
import numpy as np
import json
import matplotlib.pyplot
import pickle
from matplotlib.pyplot import imshow
from PIL import Image
from sklearn.manifold import TSNE
import sys
import numpy as np
import json
import os
from os.path import isfile, join
import keras
from keras.preprocessing import image
from keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial import distance
import rasterfairy
#import tsne_grid
import imageio



basedir = "./"
datadir = os.path.join(basedir,'data')
movie_path = os.path.join(datadir, "movie/")
image_path = os.path.join(datadir, "image/")
jsondir = os.path.join(datadir,"json/")
firstframedir = os.path.join(datadir,"ff/")

onegif = os.path.join(movie_path,os.listdir(movie_path)[0])
gif = imageio.get_reader(onegif)
gifnum = len(gif)
print(gifnum)
import shutil

for frame in range(0,2):
    for movie in os.listdir(movie_path):
        moviepath = os.path.join(movie_path,movie)
        #print(moviepath)
        filename, file_extension = os.path.splitext(moviepath)
        absfilepath = os.path.abspath(moviepath)
        image = Image.open(absfilepath)
        image.seek(1)
        basename = os.path.basename(filename)
        imgsavepath = os.path.join(firstframedir,str(frame)+"_"+basename)
        image = image.save(imgsavepath+".png",format="PNG")
    os.system("python tsne_grid.py --dir ./data/ff/ --size 10 --path ./outgrid/")
    shutil.rmtree(firstframedir)
    os.mkdir(firstframedir)
    

