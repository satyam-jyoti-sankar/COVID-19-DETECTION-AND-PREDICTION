import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50

from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.layers import Flatten , Dense, Dropout , MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
print('Jay jagannath')


from pathlib import Path
import os

# for directory path
BASE_DIR = Path(__file__).resolve().parent.parent

TEMPLATE_DIR = os.path.join(BASE_DIR,'templates')

DATASETS_PATH = os.path.join(BASE_DIR, 'data_sets')

TEST_IMAGE_PATH = os.path.join(DATASETS_PATH, 'test')

TRAIN_IMAGE_PATH = os.path.join(DATASETS_PATH, 'train')

VALIDATION_IMAGE_PATH = os.path.join(DATASETS_PATH, 'validation')


# path 
train_path  = TRAIN_IMAGE_PATH
valid_path  = VALIDATION_IMAGE_PATH
test_path   = TEST_IMAGE_PATH

# train
train_data_gen = ImageDataGenerator(preprocessing_function= preprocess_input, 
                                    zoom_range= 0.2, 
                                    horizontal_flip= True, 
                                    shear_range= 0.2,
                                    
                                    )

train = train_data_gen.flow_from_directory(directory= train_path, 
                                           target_size=(224,224))
# test
validation_data_gen = ImageDataGenerator(preprocessing_function= preprocess_input  )

valid = validation_data_gen.flow_from_directory(directory= valid_path, 
                                                target_size=(224,224))

# test
test_data_gen = ImageDataGenerator(preprocessing_function= preprocess_input )

test = train_data_gen.flow_from_directory(directory= test_path , 
                                          target_size=(224,224), 
                                          shuffle= False)

class_type = {0:'Covid',  1 : 'Normal'}

t_img , label = train.next()

# function when called will prot the images 

# def plotImages(img_arr, label):
#     for im, l in zip(img_arr,label) :
#         plt.figure(figsize= (5,5))
#         plt.imshow(im, cmap = 'gray')
#         plt.title(im.shape)
#         plt.axis = False
#         plt.show()
# plotImages(t_img, label)

res = ResNet50( input_shape=(224,224,3), include_top= False) # include_top will consider the new weights
for layer in res.layers:           # Dont Train the parameters again 
  layer.trainable = False

x = Flatten()(res.output)
x = Dense(units=2 , activation='sigmoid', name = 'predictions' )(x)

# creating our model.
model = Model(res.input, x)


# model.summary()
model.compile( optimizer= 'adam' , loss = 'categorical_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor= "val_accuracy" , min_delta= 0.01, patience= 3, verbose=1)
mc = ModelCheckpoint(filepath="bestmodel.h5", monitor="val_accuracy", verbose=1, save_best_only= True)



hist = model.fit(train, steps_per_epoch=10,
                           epochs=10, 
                           validation_data= valid ,
                           validation_steps=5, 
                           callbacks=[es,mc])

# load only the best model 
from tensorflow.keras.models import load_model
model = load_model("bestmodel.h5")

h = hist.history
#h.keys()

# checking out the accurscy of our model 
# acc = Model.evaluate(generator= test)[1] 
# print(f"The accuracy of your model is = {acc} %")

from tensorflow.keras.preprocessing import image

def load_image(link, target_size=None):
    import requests
    import shutil
    import os
    
    _, ext = os.path.splitext(link)
    
    r = requests.get(link, stream=True)
    with open('temp.' + ext, 'wb') as f:
        r.raw.decode_content = True
        shutil.copyfileobj(r.raw, f)
        
    img = image.load_img('temp.' + ext, target_size=target_size)
    return image.img_to_array(img)


def get_covid_result(img_path):
  img = load_image(img_path, target_size=(224, 224, 3))
  img = np.expand_dims(img, axis =0)

  res = class_type[np.argmax(model.predict(img))]
  print()
  x=model.predict(img)[0][0]*100
  y=model.predict(img)[0][1]*100

  # return res

  if(x==100 or y ==100):
    my_data = {'result':'Wrong Image'  }
    return my_data
  else:
    my_data = {'result':res ,'covid_chance': x , 'normal_chance':y }
    return my_data


