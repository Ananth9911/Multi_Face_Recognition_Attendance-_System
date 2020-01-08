#!/usr/bin/env python
# coding: utf-8

# In[29]:


import math
import cv2
from sklearn import neighbors
import numpy as np
import pandas as pd
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
da = os.getcwd()
try:  
    os.mkdir("{}/TrainImage".format(da))
except:
    pass
try:  
    os.mkdir("{}/models".format(da))
except:
    pass
try:  
    os.mkdir("{}/testimage".format(da))
except:
    pass
try:  
    os.mkdir("{}/unknown_pictures".format(da))
except:
    pass
names=[]


# In[30]:


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


# In[31]:



def predict(frame, knn_clf=None, model_path=None, distance_threshold=0.5):
    

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    
   
    X_img = frame
    X_face_locations = face_recognition.face_locations(X_img)

    if len(X_face_locations) == 0:
        return []

    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    return [(pred, loc) if rec else ("Not Found", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, matches)]


# In[32]:



def prediction_labels(frame, predictions):
    
    
    pil_image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        
        name = name.encode("UTF-8")

        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    del draw

    return np.asarray(pil_image)


# In[ ]:



for image_file in os.listdir("{}/testimage".format(da)):
    full_file_path = os.path.join("{}/testimage".format(da), image_file)
    if os.path.splitext(full_file_path)[1][1:] in ALLOWED_EXTENSIONS:

        print("Looking for faces in {}".format(image_file))
        frame = cv2.imread(full_file_path,-1)

        predictions = predict(frame, model_path="{}/models/trained_knn_model.clf".format(da))

        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))
            names.append(name)
        
        final_img = prediction_labels(frame, predictions)
        cv2.imshow("X",final_img)
        cv2.waitKey(0)   
    else:
        continue
cv2.destroyAllWindows()


# In[ ]:



cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 100)
rat, frame = cap.read()

while(True):
    rat, frame = cap.read()
    predictions = predict(frame, model_path="{}/models/trained_knn_model.clf".format(da))
    for name, (top, right, bottom, left) in predictions:
        print("- Found {} at ({}, {})".format(name, left, top))
        names.append(name)

        
    show_img = prediction_labels(frame, predictions)
    cv2.imshow('img',show_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()    
cv2.destroyAllWindows()


# In[ ]:


names=pd.DataFrame(names, columns=["Names"])
names= names[names.Names!="Not Found"]

attendance= pd.DataFrame(names.iloc[:,0].value_counts())
attendance.rename(index=str,columns={'Names': 'Count'},inplace=True)
attendance["Present"] =0
 


for i in range(attendance.shape[0]):
    if(attendance["Count"][i]>= 1):
        attendance["Present"][i] =1

print(attendance)
print()
attendance_final=attendance.drop(['Count'],axis=1)
attendance_final.to_csv('Attendance.csv')

