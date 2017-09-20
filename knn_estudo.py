"""
    Simple KNN study with dataset eth-80
    by: Luiz Lopes
"""

# import libs
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import dataset as ds

# SIFT descritor
sift = cv2.xfeatures2d.SIFT_create()

# Retorno do Animal
def returnAnimal(val):
    val = knn.predict(val)
    if (val == 1):
        return "VACA"
    elif (val == 2):
        return "CAVALO"
    else:
        return "Não identificado!"

# Função para teste
def testImg(image):
    img = cv2.imread("test/{}".format(image))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kps, descs = sift.detectAndCompute(gray, None)
    return [descs[0]]
    

# Descritor feito de forma manual
X2 = ds.DataSet.load_data()
Y2 = ds.DataSet.load_target()

knn = KNeighborsClassifier(n_neighbors=5, metric = "euclidean")
knn.fit(X2, Y2) 

pathTest = r"""C:\Projects\python_projects\MachineLearning\KNN\test"""
tests = os.listdir(pathTest)

for img in tests:
    print("{}|{}".format(img, returnAnimal(testImg(img))))
