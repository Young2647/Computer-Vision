import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.spatial import distance

from numpy import linalg

def DetectFeature(image) :
    siftDescriptor = cv2.SIFT_create()
    keypoints, descriptors = siftDescriptor.detectAndCompute(image, None)
    return keypoints, descriptors

def  ShowKeyPoints(image,name) :
    keypoints, descriptors = DetectFeature(image)
    imageKeypoints = cv2.drawKeypoints(image, keypoints, image)
    cv2.imwrite('keypoints_' + str(name) + '.jpg', imageKeypoints)



def MatchKeyPoints(descriptors1, descriptors2, ratio, threshold) :
    matches = []
    for i in range(len(descriptors1)) :
        smallest = np.inf
        secondSmallest = np.inf
        smallestIndex = 0
        for j in range(len(descriptors2)) :
            dist = distance.euclidean(descriptors1[i], descriptors2[j])
            if (dist < smallest) :
                smallest = dist
                smallestIndex = j
            elif (dist < secondSmallest) :
                secondSmallest = dist
        if (smallest < secondSmallest * ratio and smallest < threshold) :
            matches.append((i, smallestIndex))
    return matches

def GetCoords(keypoints1, keypoints2) :
    leftCoords = np.float32([point.pt for point in keypoints1])
    rightCoords = np.float32([point.pt for point in keypoints2])
    return leftCoords, rightCoords

def FindMatchCoords(matches, coords1, coords2) :
    targetMatchcoords = np.float32([coords1[i] for (i,_) in matches])
    originMatchcoords = np.float32([coords2[i] for (_,i) in matches])
    return (originMatchcoords, targetMatchcoords)


def FindHomography(coords1, coords2) :
    A = np.empty((2*len(coords1),9))
    for i in range(len(coords1)) :
        xi = coords2[i][0]
        yi = coords2[i][1]
        A[2*i] = [0,0,0, coords1[i][0],coords1[i][1],1, -yi*coords1[i][0],-yi*coords1[i][1],-yi*1]
        A[2*i+1] = [coords1[i][0],coords1[i][1],1, 0,0,0, -xi*coords1[i][0],-xi*coords1[i][1],-xi*1]
    u,s,v = np.linalg.svd(A)
    H = np.reshape(v[8], (3,3))
    H = (1/H[2][2]) * H
    return H 

def DoRANSAC(s, p, N, coords, threshold) :
    sampleCount = 0
    maxInlierRatio = 0
    finalH = np.empty((3,3))
    while N > sampleCount :
        leftInlier = []
        rightInlier = []
        randomSeed = random.sample(range(len(coords[0])), s)
        H = FindHomography(coords[0][randomSeed], coords[1][randomSeed])
        inlierNum = 0
        for i in range(len(coords[0])) :
            X = np.array([coords[0][i][0], coords[0][i][1],1])
            Y = np.dot(H,X.T)
            Y = (1/Y[2]) * Y
            refY = np.array([coords[1][i][0], coords[1][i][1],1])
            dist = distance.euclidean(Y, refY.T)
            if (dist < threshold) :
                inlierNum = inlierNum + 1
                leftInlier.append(coords[0][i])
                rightInlier.append(coords[1][i])
        inlierRatio = inlierNum/len(coords[0])
        if (inlierRatio > maxInlierRatio) :
            maxInlierRatio = inlierRatio
            #finalH = H
            finalH = FindHomography(leftInlier, rightInlier)
        sampleCount = sampleCount + 1
    return finalH

def test(leftImage, rightImage, H) :
    leftHeight, leftWidth = leftImage.shape[:2]
    rightHeight, rightWidth = rightImage.shape[:2]
    #translation = np.array(([1,0,3000],[0,1,2000],[0,0,1]))
    #H = np.dot(translation, H)
    temp = cv2.warpPerspective(leftImage,H,(leftWidth + 3000,leftHeight + 2000))
    cv2.imwrite('temp.jpg',temp)

def StitchImageLeft(leftImage, rightImage, H) :
    #test(leftImage, rightImage, H)
    leftHeight, leftWidth = leftImage.shape[:2]
    rightHeight, rightWidth = rightImage.shape[:2]

    
    topLeft = np.dot(H, np.array([0,0,1]))
    topLeft = (1/topLeft[2]) * topLeft

    widthOffset = abs(topLeft[0])
    heightOffset = abs(topLeft[1])

    translation = np.array(([1,0,widthOffset],[0,1,heightOffset],[0,0,1]))
    H = np.dot(translation, H)

    downRight = np.dot(H, np.array([leftWidth,leftHeight,1]))
    downRight = (1/downRight[2]) * downRight
    resultSize = (int(downRight[0]) + int(abs(widthOffset)) + rightWidth, int(downRight[1]) + int(abs(heightOffset)) + rightHeight)

    resultImage = cv2.warpPerspective(leftImage, H, resultSize)
    resultImage[int(heightOffset) : int(heightOffset) + rightHeight, int(widthOffset): int(widthOffset) + rightWidth, :] = rightImage
    height, width = np.where(resultImage[:, :, 0] != 0)
    maxHeight = max(height)
    maxWidth = max(width)
    resultImage = resultImage[0:maxHeight, 0:maxWidth,:]
    return resultImage


def StitchImageRight(leftImage, rightImage, H) :
    leftHeight, leftWidth = leftImage.shape[:2]
    rightHeight, rightWidth = rightImage.shape[:2]

    downRight = np.dot(H,np.array([rightWidth, rightHeight, 1]))
    downRight = (1/downRight[2]) * downRight
    resultSize = (rightWidth + int(abs(downRight[0])) + leftWidth, int(abs(downRight[1])) + leftHeight)
    
    downLeft = np.dot(H, np.array([0,leftHeight,1]))
    downLeft = (1/downLeft[2]) * downLeft

    resultImage = cv2.warpPerspective(rightImage, H, resultSize)
    resultImage[0: leftHeight, 0 :int(downLeft[0]),:] = leftImage[0:leftHeight, 0:int(downLeft[0]),:]
    height, width = np.where(resultImage[:, :, 0] != 0)
    maxHeight = max(height)
    maxWidth = max(width)
    resultImage = resultImage[0:maxHeight, 0:maxWidth,:]
    return resultImage
def PlotMatchCoords(leftImage,rightImage,coords) :
    for coord in coords[1] :
        match1 = cv2.circle(leftImage, (int(coord[0]), int(coord[1])),10, (255,0,0))
    for coord in coords[0] :
        match2 = cv2.circle(rightImage, (int(coord[0]), int(coord[1])),10, (255,0,0))
    cv2.imwrite("match1.jpg",match1)
    cv2.imwrite("match2.jpg",match2)

def StitchingProcess(leftImage, rightImage, direction) :
    keypoints1, descriptors1 = DetectFeature(leftImage)
    keypoints2, descriptors2 = DetectFeature(rightImage)
    leftCoords, rightCoords = GetCoords(keypoints1, keypoints2)
    matches = MatchKeyPoints(descriptors1, descriptors2, 0.75, 250)
    coords = FindMatchCoords(matches, leftCoords, rightCoords)
    #PlotMatchCoords(leftImage,rightImage,coords)
    s = 4
    p = 0.99
    threshold = 5
    N = 1000


    if (direction == 'left') :
        H = DoRANSAC(s, p, N, (coords[1], coords[0]), threshold)
        resultImage = StitchImageLeft(leftImage, rightImage, H)
    else : 
        H = DoRANSAC(s, p, N, coords, threshold)
        resultImage = StitchImageRight(leftImage, rightImage, H)
    return resultImage

img1 = cv2.imread('./images/img1.jpg')
img2 = cv2.imread('./images/img2.jpg')
img3 = cv2.imread('./images/img3.jpg')
img4 = cv2.imread('./images/img4.jpg')

img1 = cv2.resize(img1, (480,360))
img2 = cv2.resize(img2, (480,360))
img3 = cv2.resize(img3, (480,360))
img4 = cv2.resize(img4, (480,360))

#ShowKeyPoints(img1,'img1')
#ShowKeyPoints(img2,'img2')

img12 = StitchingProcess(img1, img2, 'left')
cv2.imwrite('test12.jpg', img12)
img123 = StitchingProcess(img12, img3, 'left')
img1234 = StitchingProcess(img123, img4, 'right')
cv2.imwrite('resultImage.jpg', img1234)
'''
img12 = StitchingProcess(img1, img2, 'left')
cv2.imwrite('test12.jpg', img12)
img34  = StitchingProcess(img3, img4, 'right')
cv2.imwrite('test34.jpg', img34)
img1234 = StitchingProcess(img12, img34, 'right')
cv2.imwrite('resultImage.jpg',img1234)
'''