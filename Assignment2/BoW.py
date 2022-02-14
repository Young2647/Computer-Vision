import os
import copy
import numpy as np
import pickle
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from time import *


def ReadFile(path, train_size, mode, name, class_size = 257) :
    #init
    images = []
    labels = []
    big_descriptors = []
    j = 0
    file = open(name + '.txt', 'w')
    sift = cv2.SIFT_create()


    for file_path in os.listdir(path) :
        j += 1
        if (j > class_size) :
            break

        label = file_path.split(".")[0]

        if (mode == 0) : # training set
            for i in range(0, train_size) :
                image_path = os.listdir(os.path.join(path, file_path))[i]
                image = cv2.imread(os.path.join(path,file_path,image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (128,128))
                _, descriptors = sift.detectAndCompute(image, None)
                try :
                    for descriptor in descriptors :
                        for data in descriptor :
                            file.write(str(data) + ' ')
                        file.write('\n')
                    images.append(image)
                    labels.append(label)
                    big_descriptors.append(descriptors)
                except :
                    pass
        elif (mode == 1) : # test 
            for i in range(train_size, len(os.listdir(os.path.join(path,file_path)))) :
                image_path = os.listdir(os.path.join(path, file_path))[i]
                image = cv2.imread(os.path.join(path,file_path,image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (128,128))
                _, descriptors = sift.detectAndCompute(image, None)
                try :
                    for descriptor in descriptors :
                        for data in descriptor :
                            file.write(str(data) + ' ')
                        file.write('\n')
                    images.append(image)
                    labels.append(label)
                    big_descriptors.append(descriptors)
                except :
                    pass
    print("--- image reading and feature extraction done! ---")
    return images, labels, big_descriptors

def GetAlldescriptors(name) :
    file = open(name + '.txt', "r")
    lines = file.readlines()
    descriptors = []
    for line in lines :
        descriptors_str = line.split(' ')
        descriptors.append(np.array([float(descriptors_str[i]) for i in range(len(descriptors_str) - 1)], dtype = np.float32))
    return np.array(descriptors)

def ApplyKmeans(num_clusters,max_iternum,train_size, class_size) :
    filename = "kmeans" + str(train_size) + "_class" + str(class_size) + "_k" + str(num_clusters) + "_iter" + str(max_iternum) + ".txt"
    
    if (os.path.exists(filename)) :
        file = open(filename, "rb")
        kmeans_result = pickle.load(file)
    else :
        file = open(filename, "wb")
        desciptors = GetAlldescriptors('train_descriptors' + str(train_size) + "_class" + str(class_size))
        kmeans_result = MiniBatchKMeans(n_clusters=num_clusters, max_iter=max_iternum).fit(desciptors)
        pickle.dump(kmeans_result,file)
    print("--- kmeans calculated ---")
    return kmeans_result

def GenerateHistogram(kmeans, bigdescriptors, num_clusters) :
    histogram_ = []
    for descriptors in bigdescriptors :
        cop_des = copy.deepcopy(descriptors)
        prediction = kmeans.predict(cop_des)
        histogram_.append(np.bincount(prediction, minlength=num_clusters))
    histogram_ = np.array(histogram_)
    histogram_ = StandardScaler().fit(histogram_).transform(histogram_)
    print("--- Histogram generated ---")
    return histogram_

def CalculateAccuracy(predict_result, test_labels) :
    final_accuracy = 0.0
    for i in range(len(test_labels)) :
        if (predict_result[i] == test_labels[i]) :
            final_accuracy += 1/len(test_labels)
    return final_accuracy


if __name__ == "__main__" :
    #init parameters
    train = 0
    test = 1
    train_size = 60
    class_size = 256 # how many classes to be test
    c = 0.01 #linear svm parameter C
    num_clusters = class_size * 8 # K-means   set K = num_clusters
    max_iternum = 300

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
    train_histogram = GenerateHistogram(kmeans, train_descriptors,num_clusters)
    test_histogram = GenerateHistogram(kmeans, test_descriptors,num_clusters)
    

    linear_svm = LinearSVC(C = c, max_iter=300)
    linear_svm.fit(train_histogram, train_labels)
    predict_result = linear_svm.predict(test_histogram)
    final_accuracy = CalculateAccuracy(predict_result, test_labels)
    print("--- BoW done ---")
    
    end_time = time()
    #write result to file
    file = open("BoWresult" + str(train_size) + "_class" + str(class_size) + '.txt', "w")
    file.write("kmeans max_iter = " + str(max_iternum) + '\n')
    file.write("linearSVM C = " + str(c) + "\n")
    file.write("The prediction accuracy for training size = " + str(train_size) + " is :" + str(final_accuracy) + '\n')
    file.write("Program running time is " + str(end_time - begin_time))
    

