import pandas as pd
import numpy as np

# ---------------------------- sys ----------------------------

import re
import datetime as dt

import warnings

import os
import shutil # shutil.rmtree

# ---------------------------- NLP ----------------------------

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval # iterable(string) -> object


# ------------- tensorflow & keras -----------------
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense, Activation                 #-------------FC
# from keras.layers import Conv2D, MaxPooling2D,Flatten      #-------------CNN
# from keras.layers import LSTM                              #-------------RNN
# from keras.preprocessing.image import ImageDataGenerator   #-------------Augmentation
# from keras.preprocessing.image import array_to_img, img_to_array, load_img # ㄴ flow
#
#
# from tensorflow.keras.utils import to_categorical
# from keras.callbacks import EarlyStopping, ModelCheckpoint  #------------ callback

# ----------------- fixing seed --------------------
np.random.seed(1024)
# tf.random.set_seed(1024)

# ----------------- pre -setting --------------------
warnings.filterwarnings(action='ignore')

# ===================================================
# DataFrame for searching
# ===================================================

# -------- Data load
mdf = pd.read_csv("./dataset/movies_metadata_2.csv")
ldf = pd.read_csv('./dataset/links_small.csv')
cdf = pd.read_csv("./dataset/credits.csv")
kdf = pd.read_csv("./dataset/keywords.csv")
kodf = pd.read_csv('./dataset/movie_metadata_koko.csv')


# -------- mdf pre-Processing
idx= mdf[mdf['id'].str.len()>6].index
mdf = mdf.drop(idx, axis=0)
mdf['id'] = mdf['id'].astype('int')

# -------- ldf pre-Processing
ldf = ldf.dropna()
ldf['tmdbId'] = ldf['tmdbId'].astype('int')


# -------- kodf pre-Processing
idx= kodf[kodf['id'].str.len()>6].index
kodf = kodf.drop(idx, axis=0)
kodf['id'] = kodf['id'].astype('int')


# -------- join
mldf = pd.merge(mdf, ldf, left_on='id', right_on='tmdbId')
temp = pd.merge(mldf, cdf, left_on='id', right_on='id', how = 'inner')
temp = pd.merge(temp, kodf, left_on='id', right_on='id', how = 'inner')
df = pd.merge(temp, kdf, left_on='id', right_on='id', how = 'inner')
df = df.reset_index(drop = True)


# -------- Derived variables : view_tag
df['tagline'] = df['tagline'].fillna('')
df['overview'] = df['overview'].fillna('')
df['view_tag'] = df['overview'] + df['tagline']

idx = df[df['view_tag']==''].index
df = df.drop(idx, axis = 0)

# ===================================================
# Derived variables
# ===================================================

# -------- Derived variables : RW
C = df['vote_average'].mean()
m = df['vote_count'].quantile(0.95)
def my_calc_wr(df):
    R = df['vote_average']
    v = df['vote_count']
    WR = (v / (v+m) * R + (m/(v+m)*C))
    return WR

df['wr'] = df.apply(my_calc_wr, axis=1)

# -------- Derived variables : search4 = (actor, director, key, genres)
df['cast'] = df['cast'].apply(literal_eval)            # actor
df['crew'] = df['crew'].apply(literal_eval)            # director
df['keywords'] = df['keywords'].apply(literal_eval)    # main keyword
df['genres'] = df['genres'].apply(literal_eval)        # genres


def get_director(s):
    for dict in s:
        if dict['job'] == 'Director':
            dict['name'] = dict['name'].replace(' ', '')
            return [dict['name'].lower()]  # list로 return
        # return np.nan
def get_cast(s):
    cast_list= []
    for dict in s:
        dict['name'] = dict['name'].replace(' ', '')
        cast_list.append(dict['name'].lower())
    return cast_list[:3]


df['director'] = df['crew'].apply(get_director)
df['actor'] = df['cast'].apply(get_cast)
df['key'] = df['keywords'].apply(get_cast)
# df['genres'] = df['genres']

df['search4'] = df['director'] + df['actor'] + df['key'] + df['genres']

df.to_csv('./dataset/movie_data_enko.csv', index = False)

