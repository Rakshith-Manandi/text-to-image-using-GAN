rm runtime.py txt2image_dataset.py convert_flowers_to_hd5_script.py utils.py trainer.py visualize.py 
jupyter nbconvert --to script *.ipynb

cd models
rm gan.py gan_cls.py gan_factory.py
jupyter nbconvert --to script *.ipynb
cd ..

nohup python -u runtime.py >> runtime.log & 



