from PIL import Image
import numpy as np
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
import os
from cv2 import cv2
import webbrowser

img_dir = os.getcwd()
img_width = 48
img_height = 48
results = []
accuracies = []
model_img = "predict12.jpg"
def predict_model(model_name, model_path, image, val_1, val_2):
    
    model = keras.models.load_model(model_path)
   #model.summary()
    def convert_to_array(img):
        im = cv2.imread(img)
        img = Image.fromarray(im, 'RGB')
        image = img.resize((img_width,img_height))
        return np.array(image)
    def get_model_type_name(label):
        
        if label==0:
            results.append(val_1)
            return val_1
        if label==1:
            results.append(val_2)
            return val_2
    
    def predict_model_type(file):
        print("Predicting .................................")
        ar=convert_to_array(file)
        ar=ar/255
        a=[]
        a.append(ar)
        a=np.array(a)
        score=model.predict(a,verbose=1)
        #print(score)
        label_index=np.argmax(score)
        #print(label_index)
        acc=np.max(score)
        model_type=get_model_type_name(label_index)
        #print(model_type)
        print("The predicted model  "+model_type+" with accuracy =    "+str(acc))
        accuracies.append(str(acc))
    predict_model_type(image)

predict_model("gender","gender_model.h5",model_img,"is a male", "is a female")
predict_model("glasses","glasses_model.h5",model_img,"has glasses", "has no glasses")
predict_model("smiles","smiles_model.h5",model_img,"is smiling", "is not smiling")
predict_model("makeup","makeup_model.h5",model_img,"has heavy makeup", "has not heavy makeup")
predict_model("bald","bald_model.h5",model_img,"is bald", "is not bald")
predict_model("young","young_model.h5",model_img,"is young", "is not young")
predict_model("wearhat","wearhat_model.h5",model_img,"is wearing a hat", "is not wearing a hat")
predict_model("mustache","mustache_model.h5",model_img,"has bald", "has not mustache")
predict_model("goatee","goatee_model.h5",model_img,"is goatee", "has not goatee")
predict_model("chubby","chubby_model.h5",model_img,"is chubby", "is not chubby")
predict_model("wavyhair","wavyhair_model.h5",model_img,"has wavyhair", "has not wavyhair")

html = '''
    <div style="overflow: auto;  border: 2px solid #FFF8D8;
        padding: 2px; width: 600px;" >
        <img src="{}" style="float: left;" width="150" height="150">
        <div style="padding: 10px 0px 0px 10px; overflow: auto;">           
            <h4 style="margin-left: 50px; margin-top: 2px;">{}{}</h3>
            <h4 style="margin-left: 50px; margin-top: 2px;">{}{}</h3>
            <h4 style="margin-left: 50px; margin-top: 2px;">{}{}</h3>
            <h4 style="margin-left: 50px; margin-top: 2px;">{}{}</h3>
            <h4 style="margin-left: 50px; margin-top: 2px;">{}{}</h3>
            <h4 style="margin-left: 50px; margin-top: 2px;">{}{}</h3>
            <h4 style="margin-left: 50px; margin-top: 2px;">{}{}</h3>
            <h4 style="margin-left: 50px; margin-top: 2px;">{}{}</h3>
            <h4 style="margin-left: 50px; margin-top: 2px;">{}{}</h3>
            <h4 style="margin-left: 50px; margin-top: 2px;">{}{}</h3>
            <h4 style="margin-left: 50px; margin-top: 2px;">{}{}</h3>
        </div>
    </div>
    '''.format(img_dir+"\\"+model_img,
            "gender: "+results[0],", acc: "+accuracies[0][:4],
            "glasses: "+results[1],",  acc : "+accuracies[1][:4],
            "smiling : "+results[2], ",  acc : "+accuracies[2][:4], 
            "makeup : "+results[3], ",  acc: "+accuracies[3][:4],
            "bald : "+results[4], ",  acc: "+accuracies[4][:4],
            "young : "+results[5], ",  acc: "+accuracies[5][:4],
            "wearhat : "+results[6], ",  acc: "+accuracies[6][:4],
            "mustache : "+results[7], ",  acc: "+accuracies[7][:4],
            "goatee : "+results[8], ",  acc: "+accuracies[8][:4],
            "chubby : "+results[9], ",  acc: "+accuracies[9][:4],
            "wavyhair : "+results[10], ",  acc: "+accuracies[10][:4])
path = os.path.abspath('temp.html')
url = 'file://' + path

with open(path, 'w') as f:
    f.write(html)
webbrowser.open(url)