import pandas as pd
import numpy as np
import keras
import os
import matplotlib.pyplot as plt
from cv2 import cv2 
from PIL import Image
from keras.utils import np_utils
from keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

df = pd.read_csv('dataset/list_attr_celeba.csv', encoding='utf-8')

male_list = df[(df['Male'] == 1)]['image_id'].head(500).values.tolist()
female_list = df[(df['Male'] == -1)]['image_id'].head(500).values.tolist()

glasses_list = df[(df['Eyeglasses'] == 1)]['image_id'].head(500).values.tolist()
no_glasses_list = df[(df['Eyeglasses'] == -1)]['image_id'].head(500).values.tolist()

smile_list = df[(df['Smiling'] == 1)]['image_id'].head(500).values.tolist()
nosmile_list = df[(df['Smiling'] == -1)]['image_id'].head(500).values.tolist()

makeup_list = df[(df['Heavy_Makeup'] == 1)]['image_id'].head(500).values.tolist()
nomakeup_list = df[(df['Heavy_Makeup'] == -1)]['image_id'].head(500).values.tolist()

bald_list = df[(df['Bald'] == 1)]['image_id'].head(500).values.tolist()
notbald_list = df[(df['Bald'] == -1)]['image_id'].head(500).values.tolist()

young_list = df[(df['Young'] == 1)]['image_id'].head(500).values.tolist()
notyoung_list = df[(df['Young'] == -1)]['image_id'].head(500).values.tolist()

wearhat_list = df[(df['Wearing_Hat'] == 1)]['image_id'].head(500).values.tolist()
notwearhat_list = df[(df['Wearing_Hat'] == -1)]['image_id'].head(500).values.tolist()

mustache_list = df[(df['Mustache'] == 1)]['image_id'].head(500).values.tolist()
nomustache_list = df[(df['Mustache'] == -1)]['image_id'].head(500).values.tolist()

goatee_list = df[(df['Goatee'] == 1)]['image_id'].head(500).values.tolist()
nogoatee_list = df[(df['Goatee'] == -1)]['image_id'].head(500).values.tolist()

chubby_list = df[(df['Chubby'] == 1)]['image_id'].head(500).values.tolist()
notchubby_list = df[(df['Chubby'] == -1)]['image_id'].head(500).values.tolist()

wavyhair_list = df[(df['Wavy_Hair'] == 1)]['image_id'].head(500).values.tolist()
nowavyhair_list = df[(df['Wavy_Hair'] == -1)]['image_id'].head(500).values.tolist()

img_width = 48
img_height = 48

def multimodels(model_name,x_list,y_list):

    data=[]
    labels=[]
    for val_x in x_list:
        imag=cv2.imread('img_celeba/'+val_x)
        img_from_ar = Image.fromarray(imag, 'RGB')
        resized_image = img_from_ar.resize((img_width, img_height))
        data.append(np.array(resized_image))
        labels.append(0)

    for val_y in y_list:
        imag=cv2.imread('img_celeba/'+val_y)
        img_from_ar = Image.fromarray(imag, 'RGB')
        resized_image = img_from_ar.resize((img_width, img_height))
        data.append(np.array(resized_image))
        labels.append(1)
    
    model_type=np.array(data)
    labels=np.array(labels)

    np.save(model_name+'.npy',model_type)
    np.save(model_name+'_labels.npy',labels)
    model_type=np.load(model_name+'.npy')
    labels=np.load(model_name+'_labels.npy')

    s=np.arange(model_type.shape[0])
    np.random.shuffle(s)
    model_type=model_type[s]
    labels=labels[s]

    num_classes=len(np.unique(labels))
    data_length=len(model_type)

    (x_train,x_test)=model_type[(int)(0.1*data_length):],model_type[:(int)(0.1*data_length)]
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255

    (y_train,y_test)=labels[(int)(0.1*data_length):],labels[:(int)(0.1*data_length)]
    y_train=keras.utils.to_categorical(y_train,num_classes)
    y_test=keras.utils.to_categorical(y_test,num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_width, img_height,3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))
    model.summary()
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adamax(), metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    history = model.fit(x_train,y_train,batch_size=50, epochs=90,verbose=1, validation_split=0.33, callbacks=[early_stop])
    score =  model.evaluate(x_test, y_test, verbose=1)

    print('\n', 'Test accuracy:', score[1])
    """
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
    plt.title(' model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    """
    model.save(model_name+'_model.h5')

multimodels("gender",male_list,female_list)
multimodels("glasses", glasses_list, no_glasses_list)
multimodels("smiles", smile_list, nosmile_list)
multimodels("makeup", makeup_list, nomakeup_list)
multimodels("bald", bald_list, notbald_list)
multimodels("young", notyoung_list, notyoung_list)
multimodels("wearhat", wearhat_list, notwearhat_list)
multimodels("mustache", mustache_list, nomustache_list)
multimodels("goatee", goatee_list, nogoatee_list)
multimodels("chubby", chubby_list, notchubby_list)
multimodels("wavyhair", wavyhair_list, nowavyhair_list)