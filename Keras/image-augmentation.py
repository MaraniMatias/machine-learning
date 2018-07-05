
# coding: utf-8

# # Image Augmentation
# - Check images/sample-train 
# - Check images/sample-confirm is empty
# 

# In[15]:


import numpy as np


# In[16]:


from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
from keras.applications.inception_v3 import preprocess_input


# **Check  that sample-confirm is empty**

# In[17]:


train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

jf_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True
)


# ## Check on a sample to see the image generators work in the way we expect

# In[18]:


train_generator = train_datagen.flow_from_directory('images/sample-train/',target_size=(150,150), save_to_dir='images/sample-confirm/')


# In[19]:


i=0
for batch in train_datagen.flow_from_directory('images/sample-train/', target_size=(150,150), save_to_dir='images/sample-confirm/'):
    i+=1
    if (i>10):
        break


# In[20]:


j=0
for batch in jf_datagen.flow_from_directory('images/sample-train/', target_size=(150,150), save_to_dir='images/sample-confirm/'):
    j+=1
    if ( j > 10):
        break

