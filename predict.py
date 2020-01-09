#!/usr/bin/env python
# coding: utf-8

# In[21]:


import cv2
import tensorflow.keras.backend as K
import numpy as np
from keras.models import load_model
import sys
from PIL import Image


# In[22]:
def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

model_name = 'mobile_net_7783.h5'
try :
    model = load_model(model_name, custom_objects= {'f1_m' : f1_m, 'precision' : precision_m, 'recall_m' : recall_m})
except :
    print('File ' + model_name + ' tidak ditemukan')

# In[32]:


def predict(filename):


    im=cv2.imread(filename)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    im = Image.fromarray(im)
    #im = im.transpose(Image.FLIP_LEFT_RIGHT)


    desired_size = 224
    old_size = im.size
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = im.resize(new_size, Image.ANTIALIAS)

    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))
    new_im = np.array(new_im)
    new_im = new_im.reshape(1,224,224,3)
    return model.predict(new_im/255.0)


# In[52]:

bangunan_dict = ['Aula_Barat', 'Aula_Timur', 'CC_Barat', 'CC_Timur', 'Labtek_V', 'Labtek_VI', 'Labtek_VII', 'Labtek_VIII', 'Oktagon', 'PAU', 'Perpustakaan', 'TVST']


try :
	a = predict(sys.argv[-1])
	print(bangunan_dict[a.argmax()], a.max())

except :
	print('File gambar tidak ditemukan')




# In[ ]:
