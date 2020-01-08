#!/usr/bin/env python
# coding: utf-8

# In[4]:


import math
import time
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


# In[5]:


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
   
    X = []
    y = []

    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


# In[6]:



if __name__ == "__main__":

    print("Training KNN")
    classifier = train("TrainImage",
                       model_save_path="{}/models/trained_knn_model.clf".format(da),
                       n_neighbors=2)
    print("Completed")

   

