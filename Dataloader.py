
# coding: utf-8

# In[50]:


import numpy as np
import h5py
filename = 'flowers.hdf5'
f = h5py.File(filename, 'r')

dataset_keys = [str(k) for k in f['train'].keys()]


# In[43]:


dataset_keys


# In[60]:


example_name = dataset_keys[1]


# In[61]:


example_name


# In[62]:


example = f['train'][example_name] 


# In[63]:


example


# In[64]:


right_txt = np.array(example['txt']).astype(str)


# In[65]:


right_txt


# In[66]:


right_embed = np.array(example['embeddings'], dtype=float)


# In[67]:


right_embed

