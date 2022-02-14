import os
import copy
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from time import *
from BoW import *

def UniformExtraction(image, step_size=16) :
    sift = cv2.SIFT_create()
    height = image.shape[0]
    width = image.shape[1]
    key_points = [cv2.KeyPoint(x, y, step_size) for x in range(0,width,step_size) for y in range(0,height,step_size)]
    _, descriptors = sift.compute(image, key_points)
    return descriptors


def GenerateSpmHistogram(kmeans, images, num_clusters, spm_level) :
    spm_histogram = []
    sift = cv2.SIFT_create()

    for image in images :
        single_histogram = []
        for level in range(spm_level + 1) :
            step_size = int (128/(2**level))
            y = 0
            for i in range(2**level) :
                x = 0
                for j in range(2**level) :
                    descriptors = UniformExtraction(copy.deepcopy(image[y:y+step_size,x:x+step_size]))
                    try :
                        prediction = kmeans.predict(descriptors)
                        sub_histogram = np.bincount(prediction, minlength=num_clusters)
                    except :
                        sub_histogram = np.zeros(num_clusters)
                    weight = 2 ** (level - spm_level)
                    single_histogram.append(weight *sub_histogram)
                    x += step_size
                y += step_size
        spm_histogram.append(np.array(single_histogram).ravel())
    spm_histogram = np.array(spm_histogram)
    spm_histogram = StandardScaler().fit(spm_histogram).transform(spm_histogram)
    print("--- Histogram generated ---")
    return spm_histogram



if __name__ == "__main__" :
    #init parameters
    train = 0
    test = 1
    train_size = 60
    class_size = 256 # how many classes to be test
    spm_level = 1 # spm level
    num_clusters = class_size * 4 # K-means   set K = num_clusters
    max_iternum = 300
    c = 0.01

    begin_time = time()

    train_file_name = "train_samples" + str(train_size) + "_class" + str(class_size) + ".info"
    if (os.path.exists(train_file_name)) :
        file = open(train_file_name,"rb")
        train_images, train_labels, train_descriptors = pickle.load(file)
        print("--- image reading and feature extraction done! ---")
    else :
        file = open(train_file_name,"wb")
        train_images, train_labels, train_descriptors = ReadFile("256_ObjectCategories/", train_size, train,'train_descriptors' + str(train_size) + "_class" + str(class_size), class_size)
        pickle.dump((train_images, train_labels, train_descriptors), file)
    
    test_file_name = "test_samples" + str(train_size) + "_class" + str(class_size) + ".info"
    if (os.path.exists(test_file_name)) :
        file = open(test_file_name,"rb")
        test_images, test_labels, test_descriptors = pickle.load(file)
        print("--- image reading and feature extraction done! ---")
    else :
        file = open(test_file_name,"wb")
        test_images, test_labels, test_descriptors = ReadFile("256_ObjectCategories/", train_size, test,'test_descriptors' + str(train_size) + "_class" + str(class_size), class_size)
        pickle.dump((test_images, test_labels, test_descriptors), file)
    
    

    kmeans = ApplyKmeans(num_clusters, max_iternum, train_size, class_size)
    train_histogram = GenerateSpmHistogram(kmeans, train_images,num_clusters, spm_level)
    test_histogram = GenerateSpmHistogram(kmeans, test_images,num_clusters, spm_level)
    
    linear_svm = LinearSVC(C = c, max_iter=300)
    linear_svm.fit(train_histogram, train_labels)
    predict_result = linear_svm.predict(test_histogram)
    final_accuracy = CalculateAccuracy(predict_result, test_labels)
    print("--- SPM done ---")
    
    end_time = time()
    #write result to file
    file = open("SPMresult" + str(train_size) + "_class" + str(class_size) + '.txt', "w")
    file.write("kmeans max_iter = " + str(max_iternum) + '\n')
    file.write("linearSVM C = " + str(c) + "\n")
    file.write("The prediction accuracy for training size = " + str(train_size) + " is :" + str(final_accuracy) + '\n')
    file.write("Program running time is " + str(end_time - begin_time))
    

