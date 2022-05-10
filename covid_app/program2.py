# # import tensorflow
# # from tensorflow import keras
# # # # import Pillow
# # #import image
# # import pandas as pd
# # import os
# # import shutil
# # from tensorflow.keras.layers import *
# # from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
# # from tensorflow.keras.models import Model

# # # # kanha use below code 
# # from tensorflow.keras import Sequential
# # # # from keras.models import Sequentials

# from tensorflow.keras.preprocessing import image

# # #kanha edit below code 
# # from tensorflow.keras.preprocessing.image import ImageDataGenerator
# # # # from keras.preprocessing import ImageDataGenerator

# # import numpy as np
# # from glob import glob
# # #import matplotlib as mpl
# # #from PIL import Image
# # #matplotlib.__version__
# # #import matplotlib.pyplot as plt
# # # .pyplot as yyyt
# # print('Jay jagarnath')

# # import base64
# # from io import BytesIO

# from pathlib import Path
# import os

# # for directory path
# BASE_DIR = Path(__file__).resolve().parent.parent
# TEMPLATE_DIR = os.path.join(BASE_DIR,'templates')
# DATASET_FILE_PATH = os.path.join(BASE_DIR, 'data_sets')

# # UPLOADS_IMAGE_FILE_PATH = os.path.join(DATASET_FILE_PATH, 'uploads')

# TEST_IMAGE_FILE_PATH = os.path.join(DATASET_FILE_PATH, 'test')

# TRAIN_IMAGE_FILE_PATH = os.path.join(DATASET_FILE_PATH, 'train')



# #Building architecture of the cnn
# # model=Sequential()
# # model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))
# # model.add(Conv2D(64,(3,3),activation='relu'))
# # model.add(MaxPooling2D(pool_size=(2,2)))
# # model.add(Dropout(.25))
# # model.add(Conv2D(64,(3,3),activation='relu'))
# # model.add(MaxPooling2D(pool_size=(2,2)))
# # model.add(Dropout(.25))
# # model.add(Flatten())
# # model.add(Dense(64,activation='relu'))
# # model.add(Dropout(.5))
# # model.add(Dense(2,activation='softmax'))

# # print model in table format 
# # model.summary()



# # model.compile(
# #   loss='categorical_crossentropy',
# #   optimizer='adam',
# #   metrics=['accuracy'])


# # from tensorflow.keras.preprocessing.image import ImageDataGenerator
# # train_datagen=ImageDataGenerator(rescale=1./255,
# #                                 shear_range=0.2,
# #                                 zoom_range=0.2,
# #                                 horizontal_flip=True)
# # test_datagen=ImageDataGenerator(rescale=1./255)
# # training_set=train_datagen.flow_from_directory(TRAIN_IMAGE_FILE_PATH,
# #                                               target_size=(224,224),
# #                                               batch_size=32,
# #                                               class_mode='categorical')
# # test_set=test_datagen.flow_from_directory(TEST_IMAGE_FILE_PATH,
# #                                          target_size=(224,224),
# #                                          batch_size=32,
# #                                          class_mode='categorical')

# # training_set.class_indices

# #fit the model
# #fit the model
# #fit the model
# #fit the model
# # model.fit(
# # training_set,
# # validation_data=test_set,
# # epochs=1,
# # steps_per_epoch=1,
# # validation_steps=8
# # )

# def load_image(link, target_size=None):
#     import requests
#     import shutil
#     import os
    
#     _, ext = os.path.splitext(link)
    
#     r = requests.get(link, stream=True)
#     with open('temp.' + ext, 'wb') as f:
#         r.raw.decode_content = True
#         shutil.copyfileobj(r.raw, f)
        
#     img = image.load_img('temp.' + ext, target_size=target_size)
#     return image.img_to_array(img)



# def test_cnn(img_path):
#   import matplotlib.pyplot  as plt
#   # img=image.load_img(img,target_size=(224,224,3))

#   img = load_image(img_path, target_size=(224, 224))
#   # img.shape
#   # img_array=image.img_to_array(img)
#   # img_array=img_array/255
#   # img_array=img_array.reshape((1,)+img_array.shape)
#   # XRAY_Pred=model.predict(img_array)
#   # y=XRAY_Pred.reshape(-1,)
#   # image_type_class=XRAY_Pred.argmax()
#   # if(image_type_class==1):
#   #   data = 'covid xray'
#   #   #print(data)
#   # else:
#   #   data = 'Normal Xray'
#   # x=print(data)
#  # ttt = {'mydata':data, 'myimg':plt.imshow(img),'percentage prediction':y}
#   # tt={'mydata':data,'covid_percentage_prediction':y[0],'normal_pecentage_percentage':y[1]}
#   return 'covid'


# # img_path = "https://www.princeton.edu/sites/default/files/styles/third_1x/public/images/2020/05/x-ray-image-2b_full.jpg"

# # print(result(img_path))

