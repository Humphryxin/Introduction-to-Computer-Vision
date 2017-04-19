import cv2
import numpy as np
import sys
import scipy as sp
import random
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import sys
from enum import Enum
import os


class FEATURE_TYPE(Enum):
    SIFT = 0
    SURF = 1

#This function will take an image and compute and display keypoints/features found in the image.
def Keypoints(raw_img, feature_type):  
    if feature_type == FEATURE_TYPE.SURF:
        # SURF feature
        img_surf = raw_img.copy();
        gray= cv2.cvtColor(raw_img,cv2.COLOR_BGR2GRAY);
        # Create SURF object. You can specify params here or later.
        # Here I set Hessian Threshold to 400
        surf = cv2.xfeatures2d.SURF_create(400);
        kp, des = surf.detectAndCompute(gray, None);
        img = cv2.drawKeypoints(gray, kp, img_surf, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(-1, -1, -1));
        #print('#surf keypoints in image: %d' % (len(kp)));
        #cv2.imshow('img_surf', img_surf);
        #cv2.waitKey();
        return kp, des, img_surf;
    elif feature_type == FEATURE_TYPE.SIFT:
        # SIFT feature
        img_sift = raw_img.copy();
        gray= cv2.cvtColor(raw_img,cv2.COLOR_BGR2GRAY);
        sift = cv2.xfeatures2d.SIFT_create();
        kp, des = sift.detectAndCompute(gray,None);
        cv2.drawKeypoints(gray, kp, img_sift, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(-1, -1, -1));
        #print('#sift keypoints in image: %d' % (len(kp)));
        #print('#sift descriptor in image: %d' % (len(des)))
        #cv2.imshow('img_sift', img_sift);
        #cv2.waitKey();
        return kp, des, img_sift;
    else:
        sys.exit("no feature_type defined in ", __myname__)
    

#This function will take two images, compute the keypoints for both and display matches found.
def Matching(img1, img2):
    kp1, des1, img11 = Keypoints(img1, FEATURE_TYPE.SIFT);
    kp2, des2, img22 = Keypoints(img2, FEATURE_TYPE.SIFT);
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    # 
    matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    number  = 0;
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.6*n.distance:
            matchesMask[i]=[1,0]
            number = number + 1;

    score = np.power(number/len(matches), 1/3);
    draw_params = dict(matchColor = (-1, -1, -1),
                   singlePointColor = (-1, -1, -1),
                   matchesMask = matchesMask,
                   flags = 0);
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    return (img3, score)
    #cv2.imshow('matches', img3);
    #cv2.waitKey();

img1_path = 'images/ST2MainHall4001.jpg';
img2_path = 'images/ST2MainHall4002.jpg';
img3_path = 'images/ST2MainHall4099.jpg';
img1 = cv2.imread(img1_path);
img2 = cv2.imread(img2_path);
img3 = cv2.imread(img3_path);

# keypoint display start ----------------------------------------
# kp1, des1, img11 = Keypoints(img1, FEATURE_TYPE.SIFT);
# kp2, des2, img22 = Keypoints(img1, FEATURE_TYPE.SURF);
# cv2.imwrite('sift_.png',img11)
# cv2.imwrite('surf_.png',img22)
# keypoint display end ----------------------------------------

# matching display start ---------------------------------------------------------
def full_img_path(index):
    img_path_prefix = 'images/ST2MainHall40';
    img_path_suffix = '.jpg';
    return img_path_prefix+format(index, '02d')+img_path_suffix

def readAndMatching(index1, index2):
    img1 = cv2.imread(full_img_path(index1))
    img2 = cv2.imread(full_img_path(index2))
    (img_matching, score) = Matching(img1, img2);
    cv2.imwrite('matching_'+str(index1)+'_'+str(index2)+'.png', img_matching)

def display5mathces():
    index1_1_19  = random.randint(1,19);
    index1_20_50 = random.randint(20,50);
    index1_51_78 = random.randint(51,78);
    index1_79_99 = random.randint(79,99);
    index2_1_19  = random.randint(1,19);
    index2_20_50 = random.randint(20,50);
    index2_51_78 = random.randint(51,78);
    index2_79_99 = random.randint(79,99);
    readAndMatching(index1_1_19, index2_1_19)
    readAndMatching(index1_20_50, index2_20_50)
    readAndMatching(index1_79_99, index2_79_99)
    readAndMatching(index1_20_50, index1_79_99)
    readAndMatching(index2_1_19, index2_51_78)
# display5mathces()

# img_matching = Matching(img1, img2);
# cv2.imwrite('matching_.png',img_matching)
# matching dispaly end -----------------------------------------------------------
def resizeWithCopy(arr, scale):
    if scale<1:
        sys.exit("the scale can't be less than zero or other than interger in", __myname__,)
    scale = int(scale)
    new_arr = np.zeros((arr.shape[0]*scale, arr.shape[1]*scale, 1), np.uint8)
    for i in range(0, arr.shape[0]*scale):
        for j in range(0, arr.shape[1]*scale):
            new_arr[i][j][0] = arr[int(np.floor(i/scale))][int(np.floor(j/scale))];
    return new_arr

#This function should take a directory as an argument and compute pairwise matching for all images in the directory.
def AllMatches(dir_path):
    N = 99;
    similarity_image = np.zeros((N,N), np.float64)

    for i in range(1,N+1):
        for j in range(1,N+1):
            img1 = cv2.imread(os.path.join(dir_path,full_img_path(i)))
            img2 = cv2.imread(os.path.join(dir_path,full_img_path(j)))
            (img_matching, score) = Matching(img1, img2);
            similarity_image[i-1, j-1] = score;
        print(str(i/N*100)+'%')
    #np.save('similarity_image', similarity_image)
    norm_image = np.zeros((N,N), np.uint8);
    norm_image = 255 * similarity_image;
    cv2.imshow('similarity_image', resizeWithCopy(norm_image, 6));
    cv2.waitKey()
    
# AllMatches(os.getcwd());


def Hierarchical_Clustering(dis_mat):
    Z = linkage(dis_mat);
    # calculate full dendrogram
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        truncate_mode='lastp',  # show only the last p merged clusters
        p=7,  # show only the last p merged clusters
        show_contracted=True,  # to get a distribution impression in truncated branches
    )
    plt.savefig('hieracrchical_clustering.png');
    plt.show()

similarity_image = np.load('similarity_image.npy');
Hierarchical_Clustering(similarity_image);


