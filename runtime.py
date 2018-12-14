
# coding: utf-8

# In[1]:


# import nbimporter
# import sys


# In[2]:


# sys.tracebacklimit = 0


# In[3]:


from trainer import Trainer     #trainer.ipynb file


# In[ ]:


import argparse
from PIL import Image   #This may not be used
import os    ##This may not be used
import easydict

args = easydict.EasyDict({'type': 'gan', 
                         'lr': 0.0002,
                         'l1_coef': 50,
                         'l2_coef': 100,
                         'cls': True,
                         'save_path':'./flowers_cls_test',

'inference': True,
'pre_trained_disc': 'checkpoints/flowers_cls/disc_190.pth',
'pre_trained_gen': 'checkpoints/flowers_cls/gen_190.pth',
'dataset': 'flowers', 
'split': 2,
'batch_size':64,
'num_workers':8,
'epochs':200})

trainer = Trainer(type=args.type,
                  dataset=args.dataset,
                  split=args.split,
                  lr=args.lr,
                  save_path=args.save_path,
                  l1_coef=args.l1_coef,
                  l2_coef=args.l2_coef,
                  pre_trained_disc=args.pre_trained_disc,
                  pre_trained_gen=args.pre_trained_gen,
                  batch_size=args.batch_size,
                  num_workers=args.num_workers,
                  epochs=args.epochs
                  )

if not args.inference:
    trainer.train(args.cls)
else:
    trainer.predict()

