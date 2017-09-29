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
from sklearn.metrics import accuracy_score
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

def testImgGCH(image):
    img = cv2.imread("test/{}".format(image))
    rows, cols, channels = img.shape
    histogram_b = np.zeros(4)
    histogram_g = np.zeros(4)
    histogram_r = np.zeros(4)

    for row in range(rows):
        for col in range(cols):
            pixel_b = img[row, col, 0] / 64
            pixel_g = img[row, col, 1] / 64
            pixel_r = img[row, col, 2] / 64
            histogram_b[int(pixel_b)] = histogram_b[int(pixel_b)] + 1
            histogram_g[int(pixel_g)] = histogram_g[int(pixel_g)] + 1
            histogram_r[int(pixel_r)] = histogram_r[int(pixel_r)] + 1

    vector_features = np.append(histogram_b, histogram_g)
    return [np.append(vector_features, histogram_r)]
    

# Dataset feito de forma manual
X2 = ds.DataSet.dataset_load_horse_cow()
Y2 = ds.DataSet.load_target()

knn = KNeighborsClassifier(n_neighbors=5, metric = "euclidean")
knn.fit(X2, Y2) 

pathTest = r"""C:\Projects\python_projects\MachineLearning\KNN\test"""
tests = os.listdir(pathTest)

for img in tests:
    print("{}|{}".format(img, returnAnimal(testImgGCH(img))))
