
# coding: utf-8

# In[2]:


import sys
sys.path.insert(0, "/datasets/home/23/223/rmanandi/text-to-image-using-GAN/")


# In[3]:


#import nbimporter


# In[4]:


from models import gan, gan_cls


# In[6]:


class gan_factory(object):

    @staticmethod
    def generator_factory(type):
        if type == 'gan':
            return gan_cls.generator()
        elif type == 'vanilla_gan':
            return gan.generator()

