
# coding: utf-8

# In[6]:


import sys
sys.path.insert(0, "/datasets/home/23/223/rmanandi/text-to-image-using-GAN/")


# In[7]:


import nbimporter


# In[8]:


from models import gan, gan_cls


# In[9]:


class gan_factory(object):

    @staticmethod
    def generator_factory(type):
        if type == 'gan':
            return gan_cls.generator()
        elif type == 'vanilla_gan':
            return gan.generator()
        
    @staticmethod
    def discriminator_factory(type):
        if type == 'gan':
            return gan_cls.discriminator()
        elif type == 'vanilla_gan':
            return gan.discriminator()
    

