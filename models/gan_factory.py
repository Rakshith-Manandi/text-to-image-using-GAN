
# coding: utf-8

# In[6]:


# import sys
# sys.path.insert(0, "/datasets/home/30/930/dmajumde/text-to-image-using-GAN/")


# In[7]:


# import nbimporter


# In[8]:


#from models import gan,gan_cls
import gan_cls

# In[4]:


class gan_factory(object):

    @staticmethod
    def generator_factory(type):
        if type == 'gan':
            return gan_cls.generator()

    @staticmethod
    def discriminator_factory(type):
        if type == 'gan':
            return gan_cls.discriminator()

