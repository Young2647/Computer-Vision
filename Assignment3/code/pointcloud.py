import numpy
import open3d as o3d
import cv2
import numpy as np
import os
fx = 5.1885790117450188e+02
fy = 5.1946961112127485e+02 
cx = 3.2558244941119034e+02
cy = 2.5373616633400465e+02

intrinsics = [
   [fx, 0., cx, 0.], 
   [0., fy, cy, 0.],
   [0., 0., 1., 0.],
   [0., 0., 0., 1.]
]

def build_point_cloud(image, depth, K) :
    height, width = image.shape[:2]
    X = np.zeros((width, height))
    Y = np.zeros((width, height))
    Z = depth
    
    #u,v is image coords
    u = range(0, width)
    v = range(0, height)
    u,v = np.meshgrid(u, v)
    u = u.astype(np.float)
    v = v.astype(np.float)
    
    #calculate coords in camera coordinate
    X = Z * (u - cx)/fx
    Y = Z * (v - cy)/fy

    points = np.zeros((6, width * height))
    points[0] = np.ravel(X)
    points[1] = np.ravel(Y)
    points[2] = np.ravel(Z)
    points[3] = np.ravel(image[:, :, 0])
    points[4] = np.ravel(image[:, :, 1])
    points[5] = np.ravel(image[:, :, 2])
    points = points.T.tolist()
    return points

def write_to_ply(points, index) :
    point_cloud = []
    for point in points :
        point_cloud.append("%f %f %f %d %d %d 0\n" % (point[0], point[1], point[2], point[3], point[4], point[5]))
    cloud_path = "point_cloud"
    if not os.path.exists(cloud_path) :
        os.mkdir(cloud_path) 
    cloud_file = open("./{}/point_cloud_{}.ply".format(cloud_path, index),"w")
    cloud_file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar blue
    property uchar green
    property uchar red
    property uchar alpha
    end_header
    %s
    ''' % (len(point_cloud), "".join(point_cloud)))
    cloud_file.close()
    return "./{}/point_cloud_{}.ply".format(cloud_path, index)


if __name__ == "__main__" : 
    index = '000590'
    image = cv2.imread("./nyuv2/test/{}_color.jpg".format(index))
    depth = cv2.imread("./test_result/img_{}.png".format(index),-1)
    #depth = cv2.imread("./nyuv2/test/{}_depth.png".format(index),-1)
    #depth = cv2.resize(depth, (304,228))
    #image processing
    image = cv2.resize(image, (304,228))
    depth = depth.astype(np.float)/ 1000.
    points = build_point_cloud(image, depth, K= intrinsics)
    cloud_file_name = write_to_ply(points, index)
    cloud = o3d.read_point_cloud(cloud_file_name)
    o3d.draw_geometries([cloud])