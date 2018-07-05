
# coding: utf-8

# # Using VGG16

# In[1]:


import numpy as np
from keras.applications import vgg16
from keras.preprocessing import image


# In[2]:


model = vgg16.VGG16(weights='imagenet')


# In[3]:


img = image.load_img('images/spoon.jpeg',target_size=(224,224))
img


# In[4]:


# Convert to Numpy array
arr = image.img_to_array(img)
arr.shape


# In[5]:


# expand dimension
arr = np.expand_dims(arr, axis=0)
arr.shape


# In[6]:


# preprocessing
arr = vgg16.preprocess_input(arr)
arr


# In[7]:


# predict
preds = model.predict(arr)
preds


# In[8]:


# predictions for top 5
vgg16.decode_predictions(preds, top=5)


# ## Test using another image

# In[9]:


img2 = image.load_img('images/fly.jpeg',target_size=(224,224))
img2


# In[10]:


arr2 = image.img_to_array(img2)
arr2 = np.expand_dims(arr2,axis=0)
arr2 = vgg16.preprocess_input(arr2)
preds2 = model.predict(arr2)
vgg16.decode_predictions(preds2, top=5)

