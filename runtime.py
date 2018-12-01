
# coding: utf-8

# In[31]:


import sys
sys.path.insert(0, "/datasets/home/23/223/rmanandi/text-to-image-using-GAN/")
import nbimporter


# In[32]:


# from trainer import Trainer     #trainer.ipynb file
import trainer
import argparse
from PIL import Image   #This may not be used
import os    ##This may not be used
import easydict


# In[33]:


args = easydict.EasyDict({'type': 'gan', 
                         'lr': 0.0002,
                         'l1_coef': 50,
                         'l2_coef': 100,
                         'diter': 5,
                         'cls': False,
                         'vis_screen': 'gan',
                         'save_path':'',

'inference': False,
'pre_trained_disc': None,
'pre_trained_gen': None,
'dataset': 'flowers', 
'split': 0,
'batch_size':64,
'num_workers':8,
'epochs':200})

trainer = trainer.Trainer(type=args.type,
                  dataset=args.dataset,
                  split=args.split,
                  lr=args.lr,
                  diter=args.diter,   #This may not be used 
                  vis_screen=args.vis_screen,
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


# In[9]:


parser = argparse.ArgumentParser()
parser.add_argument("--type", default='gan')
parser.add_argument("--lr", default=0.0002, type=float)
parser.add_argument("--l1_coef", default=50, type=float)
parser.add_argument("--l2_coef", default=100, type=float)
parser.add_argument("--diter", default=5, type=int) #This may not be used
parser.add_argument("--cls", default=False, action='store_true')
parser.add_argument("--vis_screen", default='gan')
parser.add_argument("--save_path", default='')
parser.add_argument("--inference", default=False, action='store_true')
parser.add_argument('--pre_trained_disc', default=None)
parser.add_argument('--pre_trained_gen', default=None)
parser.add_argument('--dataset', default='flowers')
parser.add_argument('--split', default=0, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--epochs', default=200, type=int)
args = parser.parse_args(argv[1:])

trainer = trainer.Trainer(type=args.type,
                  dataset=args.dataset,
                  split=args.split,
                  lr=args.lr,
                  diter=args.diter,   #This may not be used 
                  vis_screen=args.vis_screen,
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

