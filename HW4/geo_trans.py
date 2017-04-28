import cv2
import numpy as np
import sys
import math
import scipy as sp
import random
from matplotlib import pyplot as plt
import sys
from enum import Enum
import os
from skimage.transform import warp, AffineTransform
from skimage.measure import ransac
from pprint import pprint

np.set_printoptions(suppress=True)
np.set_printoptions(precision=6)

class FEATURE_TYPE(Enum):
    SIFT = 0
    SURF = 1
class MATCHING_TYPE(Enum):
    TWO_WAY_MATCHING = True;
    ONE_WAY_MATCHING = False;
class TRANSFORMATION_TYPE(Enum):
    AFFINE = 0;
    HOMOGRAPHY = 1;

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
    

# This function will take two images, compute the keypoints for both and display matches found.
# The current score estimation is based on the percentage of kept match among all the matches
def Matching(img1, img2, FEATURE_OPTION, MATCHING_OPTION):
    kp1, des1, img11 = Keypoints(img1, FEATURE_OPTION);
    kp2, des2, img22 = Keypoints(img2, FEATURE_OPTION);
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    number = 0;
    good = [];
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        # cv::DMatch Class Reference
        # DMatch (int _queryIdx, int _trainIdx, int _imgIdx, float _distance)
        # print(m.imgIdx, m.queryIdx, m.trainIdx) 
        if m.distance < 0.6*n.distance:
            good.append(m);
            matchesMask[i]=[1,0]
            number = number + 1;
    
    if MATCHING_OPTION == MATCHING_TYPE.TWO_WAY_MATCHING:
        good = [];
        matches2 = flann.knnMatch(des2, des1, k=2)
        for i,(m,n) in enumerate(matches2):
            if m.distance < 0.6*n.distance:
                if matchesMask[m.trainIdx] == [1,0]:
                    matchesMask[m.trainIdx] = [1,1]
                ## The following operation will change the value of m
                m_copy = m;
                temp = m_copy.trainIdx;
                m_copy.trainIdx = m_copy.queryIdx;
                m_copy.queryIdx = temp;
                good.append(m_copy);
        matchesMask_copy = matchesMask.copy();
        matchesMask = [[0,0] if x==[1,0] else x for x in matchesMask_copy]
        matchesMask = [[1,0] if x==[1,1] else x for x in matchesMask]
    score = np.power(number/len(matches), 1/3);
    # For now just set MIN_MATCH_COUNT to 0
    MIN_MATCH_COUNT = 0;
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    draw_params = dict(matchColor = (-1, -1, -1),
                   singlePointColor = (-1, -1, -1),
                   matchesMask = matchesMask,
                   flags = 0);
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    return (img3, score, src_pts, dst_pts)

def generate_img_path(idx):
    img_path_pre = '2010_06_10/IMG_';
    img_path_post = '.JPG';
    return img_path_pre+str(idx)+img_path_post;
# Choose a descriptor and demonstrate matching between three image pairs. 
def Question1(num_of_pairs):
    FEATURE_OPTION = FEATURE_TYPE.SIFT;
    MATCHING_OPTION = MATCHING_TYPE.TWO_WAY_MATCHING;
    # sample random number without replacemetn
    img_idx_list = random.sample(range(1188,1219), num_of_pairs*2);
    for i in range(math.floor(len(img_idx_list)/2)):
        img1_path = generate_img_path(img_idx_list[2*i]);
        img2_path = generate_img_path(img_idx_list[2*i+1]);
        img1 = cv2.imread(img1_path);
        img2 = cv2.imread(img2_path);
        (matching_visualize, score, src_pts, dst_pts) = Matching(img1, img2, FEATURE_OPTION, MATCHING_OPTION);
        cv2.imwrite('matching_'+str(img_idx_list[2*i])+'_'+str(img_idx_list[2*i+1])+'.png', matching_visualize)
#Question1(3);


def findAffineTransformation(src_pts, dst_pts):
    # estimate affine transform model using all coordinates
    # before reshape (1162, 1, 2) (1162, 1, 2)
    # after reshape (1162, 2) (1162, 2)
    src_pts = src_pts.reshape(-1,2);
    dst_pts = dst_pts.reshape(-1,2);
    model = AffineTransform()
    model.estimate(src_pts, dst_pts)

    # robustly estimate affine transform model with RANSAC
    M_affine, inliers = ransac((src_pts, dst_pts), AffineTransform, min_samples=4, residual_threshold=2, max_trials=200)
    return M_affine.params, inliers
        
def Question2(num_of_pairs):
    FEATURE_OPTION = FEATURE_TYPE.SIFT;
    MATCHING_OPTION = MATCHING_TYPE.TWO_WAY_MATCHING;
    # sample random number without replacemetn
    img_idx_list = random.sample(range(1188,1219), num_of_pairs*2);
    for i in range(math.floor(len(img_idx_list)/2)):
        img1_path = generate_img_path(img_idx_list[2*i]);
        img2_path = generate_img_path(img_idx_list[2*i+1]);
        img1 = cv2.imread(img1_path);
        img2 = cv2.imread(img2_path);
        (matching_visualize, score, src_pts, dst_pts) = Matching(img1, img2, FEATURE_OPTION, MATCHING_OPTION);
        M_homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # M_affine = cv2.getAffineTransform(src_pts, dst_pts);
        M_affine, inliers = findAffineTransformation(src_pts, dst_pts);
        print(str(img_idx_list[2*i]), str(img_idx_list[2*i+1]))
        print(M_homography)
        print(M_affine)

# Question2(3);

def addBorder2Image(src, border_size, translation_rows, translation_cols):
    border_size= border_size*5;
    rows,cols,depth = src.shape;
    dst = np.zeros((rows+2*border_size+np.abs(translation_rows), cols+2*border_size+np.abs(translation_cols), depth), np.uint8);
    if translation_rows>0:
        new_rows1 = border_size+np.abs(translation_rows);
        new_rows2 = rows+border_size+np.abs(translation_rows);
    else:
        new_rows1 = border_size;
        new_rows2 = rows+border_size;
    if translation_cols>0:
        new_cols1 = border_size+np.abs(translation_cols);
        new_cols2 = cols+border_size+np.abs(translation_cols);
    else:
        new_cols1 = border_size;
        new_cols2 = cols+border_size;
    dst[new_rows1:new_rows2, new_cols1:new_cols2] = src;
    return dst, new_rows1, new_cols1
    
def test_fun():
    n = random.sample(range(1188,1219), 1);
    img_path = generate_img_path(n[0]);
    img = cv2.imread(img_path);
    rows = img.shape[0];
    cols = img.shape[1];
    rotation_displacement = math.floor((np.sqrt(rows*rows+cols*cols)-min(rows,cols))/2);
    dst,new_rows1, new_cols1 = addBorder2Image(img, rotation_displacement, 100, 100)
    cv2.imwrite('test_border.png', dst);

#test_fun()
def cropBlackBorder(src):
    img = np.sum(src,axis = 2);
    mask = img > 0;
    # Coordinates of non-black pixels.
    coords = np.argwhere(mask)
    # Bounding box of non-black pixels.
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top
    # Get the contents of the bounding box.
    cropped = src[x0:x1, y0:y1];
    return cropped

def test_crop():
    n = random.sample(range(1188,1219), 1);
    img_path = 'dst_homo.png';
    img = cv2.imread(img_path);
    rows = img.shape[0];
    cols = img.shape[1];
    dst = cropBlackBorder(img);
    cv2.imwrite('test_crop.png', dst);

#test_crop();    

# put img1 to img2
def image_stiching(img2, img1, FEATURE_OPTION, MATCHING_OPTION):
    (matching_visualize, score, src_pts, dst_pts) = Matching(img1, img2, FEATURE_OPTION, MATCHING_OPTION);
    M_homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # Option 1: Just 3 points
    # M_affine = cv2.getAffineTransform(src_pts, dst_pts);
    # Optoin 2: Potentially us all points
    M_affine, inliner = findAffineTransformation(src_pts, dst_pts);
    rows,cols = img1.shape[:2];
    translation_cols,translation_rows = M_affine[0:2,2];
    translation_rows = math.floor(translation_rows);
    translation_cols = math.floor(translation_cols);
    rotation_displacement = max(math.floor(max(img2.shape)/3) , math.floor((np.sqrt(rows*rows+cols*cols)-min(rows,cols))/2));
    dst, new_rows, new_cols = addBorder2Image(img1, rotation_displacement, translation_rows, translation_cols)
    #dst, new_rows, new_cols = addBorder2Image(img1, 0, 0, 0)
    rows,cols = dst.shape[:2];
    T1 = np.array([[1,0,new_cols],[0,1,new_rows],[0,0,1]], np.float64);
    #print(M_affine)
    M_affine = np.matmul(T1, np.matmul(M_affine,np.linalg.inv(T1)));
    #print(M_homography)
    M_homography = np.matmul(T1,np.matmul(M_homography,np.linalg.inv(T1)));
    ## warpAffine : (width, height)
    dst1_affine = cv2.warpAffine(dst, M_affine[:2,:], (cols,rows))
    dst1_homo = cv2.warpPerspective(dst, M_homography, (cols,rows))
    # Now create a mask of logo and create its inverse mask also
    roi = dst1_affine[new_rows:new_rows+img2.shape[0],new_cols:new_cols+img2.shape[1]];
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask = mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2, img2, mask = mask)
    # Put logo in ROI and modify the main image
    img2_dst = cv2.add(img1_bg,img2_fg);
    dst1_affine[new_rows:new_rows+img2.shape[0],new_cols:new_cols+img2.shape[1]] = img2_dst;
    dst1_homo[new_rows:new_rows+img2.shape[0],new_cols:new_cols+img2.shape[1]] = img2_dst;
    dst_homo = cropBlackBorder(dst1_homo);
    dst_affine = cropBlackBorder(dst1_affine);
    return dst_affine, dst_homo
    
# image stitching
def Question3(num_of_pairs):
    FEATURE_OPTION = FEATURE_TYPE.SIFT;
    MATCHING_OPTION = MATCHING_TYPE.TWO_WAY_MATCHING;
    # sample random number without replacemetn
    img_idx_list = random.sample(range(1188,1219), num_of_pairs*2);
    for i in range(math.floor(len(img_idx_list)/2)):
        img1_path = generate_img_path(img_idx_list[2*i]);
        img2_path = generate_img_path(img_idx_list[2*i+1]);
        img1 = cv2.imread(img1_path);
        img2 = cv2.imread(img2_path);
        dst_affine,dst_homo = image_stiching(img1, img2, FEATURE_OPTION, MATCHING_OPTION);
        cv2.imwrite('stiching_affine'+str(img_idx_list[2*i])+'_'+str(img_idx_list[2*i+1])+'.png', dst_affine)
        cv2.imwrite('stiching_homo'+str(img_idx_list[2*i])+'_'+str(img_idx_list[2*i+1])+'.png', dst_homo)

#Question3(3);

        
        
def Question4(num_of_imgs):
    FEATURE_OPTION = FEATURE_TYPE.SIFT;
    MATCHING_OPTION = MATCHING_TYPE.TWO_WAY_MATCHING;
    # sample random number without replacemetn
    img_idx_list = random.sample(range(1188,1219), num_of_imgs);
    img_path = generate_img_path(img_idx_list[0]);
    dst_affine = cv2.imread(img_path);
    dst_homo = dst_affine.copy()
    name_list = str(img_idx_list[0]);
    for i in range(math.floor(len(img_idx_list))-1):
        img2_path = generate_img_path(img_idx_list[i+1]);
        img2 = cv2.imread(img2_path);
        dst_affine, temp = image_stiching(dst_affine, img2, FEATURE_OPTION, MATCHING_OPTION);
        temp, dst_homo = image_stiching(dst_homo, img2, FEATURE_OPTION, MATCHING_OPTION);
        name_list = name_list + '_' + str(img_idx_list[i+1]);
    cv2.imwrite('pano_affine_' + name_list +'.png', dst_affine)
    cv2.imwrite('pano_homo_' + name_list +'.png', dst_homo)
Question4(5)

