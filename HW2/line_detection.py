import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

## folder name construction
Images_folder = "ST2MainHall4";
Images_prefix = "ST2MainHall40";
Images_format = ".jpg";
N_img = 5;
# random label: used for coding.
img_label = np.random.randint(10,99, N_img);
# we choose the label as follow:
img_label = np.array([10, 26, 46, 66, 95]);
# canny parameters
minVal = 100;
maxVal = 150;
# hough parameters
minVote = 230;
## houghp parameters
minLineLength = 20
maxLineGap = 100
## filter parameters
minGrad = 0.03;
maxOriDiff = 0.2;
## filter out fake vanishing points
maxDis = 20;# maximal distance to be considererd to be nearby
minNum = 11; # minimal number of points nearby
maxDis_center = 30;
### ---------- from this point, no constant variables are declared.

### ------ function definitions

def hough_visualize(hough_img, lines):
    for line in lines:
        rho, theta = line[0];
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 2000*(-b))
        y1 = int(y0 + 2000*(a))
        x2 = int(x0 - 2000*(-b))
        y2 = int(y0 - 2000*(a))
        cv2.line(hough_img,(x1,y1),(x2,y2),(0,0,255),2)
    return hough_img

def check_index(y, x, shape):
    if (y<0) or (x < 0):
        return False;
    elif (y>=shape[1]) or (x>=shape[0]):
        return False;
    else:
        return True;
    
def filter_linesp(linesp, bgr_img, minGrad, maxOriDiff):
    filterp_img = bgr_img.copy();
    for linep in linesp:
        x0,y0,x1,y1 = linep[0];
        if (x0 > x1):
            x_t = x1;
            x1 = x0;
            x0 = x_t;
        if (x1 - x0)==0:
            for y in range(y0, y1, np.sign(y1-y0)):
                filterp_img[y, x0] = [0,255 ,0];
            continue;
        deltax = float(x1 - x0)
        deltay = float(y1 - y0)
        deltaerr = abs(deltay / deltax)
        error = deltaerr - 0.5;
        y = y0;
        for x in list(range(x0,x1+1)):
            filterp_img[y, x] = [0,255 ,0];
            error = error + deltaerr;
            if error >= 0.5:
                y = y + np.sign(y1-y0);
                error = error - 1.0
    #cv2.imwrite("images/filterp_img.jpg", filterp_img);    
    return filterp_img;  

def gradient_img(bgr_img):
    # compute kernels for computing image derivatives
    kernel_x = np.array([[-1.0, 0.0, 1.0]])/2.0;
    kernel_y = np.array([[-1.0], [0.0], [1.0]])/2.0;
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY);
    gaussian_gray2 = cv2.GaussianBlur(gray_img, (13, 13), 2);
    x_gaussian_gray2 = cv2.filter2D(gaussian_gray2, cv2.CV_32F, kernel_x)
    y_gaussian_gray2 = cv2.filter2D(gaussian_gray2, cv2.CV_32F, kernel_y)
    mag_gaussian_gray2 = np.sqrt(np.add(np.multiply(x_gaussian_gray2,x_gaussian_gray2), np.multiply(y_gaussian_gray2, y_gaussian_gray2)))
    ori_gaussian_gray2 = np.arctan2(y_gaussian_gray2,x_gaussian_gray2);
    return (mag_gaussian_gray2, ori_gaussian_gray2)


def filter_lines(lines, bgr_img, minGrad, maxOriDiff):
    filter_img = bgr_img.copy();
    ##
    (mag_grad_img, ori_grad_img) = gradient_img(bgr_img);
    
    
    for line in lines:
        rho, theta = line[0];
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 2500*(-b))
        y1 = int(y0 + 2500*(a))
        x2 = int(x0 - 2500*(-b))
        y2 = int(y0 - 2500*(a))
        
        if (np.absolute(theta) < 0.8 and (x1 != x2)):
            if (y2 < y1):
                y_t = y1;
                y1 = y2;
                y2 =y_t;
            deltax = float(x2 - x1)
            deltay = float(y2 - y1)
            deltaerr = abs(deltax / deltay)
            error = deltaerr - 0.5;
            x = x1;
            #print(deltaerr)
            for y in range(y1,y2+1):
                if check_index(x, y, filter_img.shape):
                    filter_img[y, x] = [0,255 ,0];
                    if (mag_grad_img[y, x] < minGrad):
                        filter_img[y, x] = [0, 0, 255];
                    if ((np.absolute(np.sin(ori_grad_img[y, x])-np.sin(theta))) > maxOriDiff):
                            filter_img[y, x] = [0,0,255];
                error = error + deltaerr;
                if error >= 0.5:
                    x = x + np.sign(x2-x1);
                    error = error - 1.0
        else:
            if (x2 < x1):
                x_t = x1;
                x1 = x2;
                x2 = x_t;
            if (x2 - x1)==0:
                for y in range(y1, y2, np.sign(y2 - y1)):
                    if check_index(x1, y, filter_img.shape):
                        filter_img[y, x1] = [0,255,0];
                        if (mag_grad_img[y, x1] < minGrad):
                            filter_img[y, x1] = [0, 0, 255];
                        if ((np.absolute(np.sin(ori_grad_img[y, x1])-np.sin(theta))) > maxOriDiff):
                            filter_img[y, x1] = [0,0,255];
                continue;

            deltax = float(x2 - x1)
            deltay = float(y2 - y1)
            deltaerr = abs(deltay / deltax)
            error = deltaerr - 0.5;
            y = y1;
            #print(deltaerr)
            for x in range(x1,x2+1):
                if check_index(x, y, filter_img.shape):
                    filter_img[y, x] = [0,255 ,0];
                    if (mag_grad_img[y, x] < minGrad):
                        filter_img[y, x] = [0, 0, 255];
                    if ((np.absolute(np.sin(ori_grad_img[y, x])-np.sin(theta))) > maxOriDiff):
                            filter_img[y, x] = [0,0,255];
                error = error + deltaerr;
                if error >= 0.5:
                    y = y + np.sign(y2-y1);
                    error = error - 1.0
    return filter_img;

def detect_allpair_van(lines, bgr_img):
    van_img = bgr_img.copy();
    first_flag = False;
    for line1 in lines:
        rho, theta = line1[0];
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 2500*(-b))
        y1 = int(y0 + 2500*(a))
        x2 = int(x0 - 2500*(-b))
        y2 = int(y0 - 2500*(a))
        cv2.line(van_img,(x1,y1),(x2,y2),(0, 255, 0),2)
        
        for line2 in lines:
            rho1, theta1= line2[0];
            a1 = np.cos(theta1)
            b1 = np.sin(theta1)
            if (np.absolute(b*a1-a*b1) < 0.1):
                continue;
            inter_x =((-rho)*b1- b*(-rho1))/(b*a1-a*b1);
            inter_y =(a*(-rho1)-(-rho)*a1)/(b*a1-a*b1);
            #
            cv2.circle(van_img, (inter_x, inter_y), 30, [0, 0, 255], 4)
            if first_flag==False:
                first_flag = True;
                inter_points = np.array([[inter_x, inter_y, rho, rho1]]);
            elif check_index(inter_x, inter_y, van_img.shape):
                inter_points = np.append(inter_points, [[inter_x, inter_y, theta, theta1]], axis=0);
    return (van_img, inter_points)

def length_2_index(length_van_points):
    index_van_points = np.array([0], dtype=np.int);
    temp_index = 0;
    for i in range(0, length_van_points.shape[0]):
        temp_index = temp_index + length_van_points[i];
        index_van_points= np.append(index_van_points, temp_index);
    return index_van_points;

def filter_fake_van(inter_points, maxDis, minNum):
    remain_pionts=  inter_points.copy();
    van_points = np.array([]);
    length_van_points = np.array([],dtype=np.int);
    first_Flag = True;
    # the vanishing points group(with a minimal size of 5) which are spatially close in image are potential VPs
    for point in inter_points:
        if point in remain_pionts:
            Count = 0;
            temp_arr = np.array([], dtype=np.int);
            for i in range(0,remain_pionts.shape[0]):
                if (np.sqrt((point[0]-remain_pionts[i,0])*(point[0]-remain_pionts[i,0]) + (point[1]-remain_pionts[i,1])*(point[1]-remain_pionts[i,1])) < maxDis):
                    Count = Count+1;
                    temp_arr = np.append(temp_arr, i);
            if Count >  minNum:
                for i in temp_arr:
                    if (first_Flag == True):
                        van_points = np.array([remain_pionts[i]])
                        first_Flag = False;
                    else:
                        van_points = np.append(van_points, [remain_pionts[i]], axis=0);
                length_van_points = np.append(length_van_points, Count);
                remain_pionts = np.delete(remain_pionts, temp_arr, 0)
                
    # the vanishing points group which have a variaty of orientations are candidate VPs
    index_van_points = length_2_index(length_van_points);
    temp_van_points = van_points.copy();
    
    delete_index = np.array([],dtype= np.int);
    for i in range(index_van_points.shape[0]-1):
        N = length_van_points[i];
        orien_array = np.array([]);
        for j in range(index_van_points[i], index_van_points[i+1]):
            orien_array = np.append(orien_array, van_points[j,2:]);
        orien_array = np.unique(orien_array);
        if (orien_array.shape[0]*(orien_array.shape[0]-1)/2 < N-1):
             delete_index = np.append(delete_index, range(index_van_points[i], index_van_points[i+1]));     
    temp_van_points = np.delete(temp_van_points, delete_index, 0)
    
    return temp_van_points;

def num_cluster(centers, maxDis):
    remain_pionts = centers.copy();
    # the vanishing points group(with a minimal size of 5) which are spatially close in image are potential VPs
    Count = 0;
    for point in centers:
        if point in remain_pionts:
            Count = Count + 1;
            temp_arr = np.array([], dtype=np.int);
            for i in range(0,remain_pionts.shape[0]):
                if (np.sqrt((point[0]-remain_pionts[i,0])*(point[0]-remain_pionts[i,0]) + (point[1]-remain_pionts[i,1])*(point[1]-remain_pionts[i,1])) < maxDis):
                    temp_arr = np.append(temp_arr, i);
            remain_pionts = np.delete(remain_pionts, temp_arr, 0);
    return Count;
                

def detect_van(inter_points, bgr_img, maxDis, minNum, maxDis_center):
    van_img = bgr_img.copy();
    #
    van_points = filter_fake_van(inter_points, maxDis, minNum);
    for line1 in lines:
        rho, theta = line1[0];
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 2500*(-b))
        y1 = int(y0 + 2500*(a))
        x2 = int(x0 - 2500*(-b))
        y2 = int(y0 - 2500*(a))
        cv2.line(van_img,(x1,y1),(x2,y2),(0, 255, 0),2)
        
    # define criteria and apply kmeans()
    if van_points.shape[0] > 0:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, centers=cv2.kmeans(np.float32(van_points[:,:2]),5,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS);
        K = num_cluster(centers, maxDis_center);
        ret, label, centers=cv2.kmeans(np.float32(van_points[:,:2]),K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS);
        for center in centers:
            cv2.circle(van_img, (center[0], center[1]), 30, [0, 0, 255], 4);
    # apply least squares method
    # the mean of x and y
    
    
    return van_img
### ----- end function definition

target = open("images/inter_points.txt", 'w');
for i in range(N_img):
    img_path = os.path.join(os.getcwd(),Images_folder, Images_prefix + str(img_label[i]) + Images_format);
    bgr_img = cv2.imread(img_path);
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY);
    
    if bgr_img is None:
        print(img_path+" doesn't exist!")
        break;
    ## canny edge detection: input, color_img; output, edge_img
    ## initial 100, 200
    edges_img = cv2.Canny(gray_img, minVal, maxVal, minNum);
    
    ## standard hough tranform
    lines = cv2.HoughLines(edges_img, 1, np.pi/180, minVote);
    ## probabilistic hough transform
    ## Just we have to decrease the threshold.
    
    linesp = cv2.HoughLinesP(edges_img,1,np.pi/180,100,minLineLength,maxLineGap);
    ## Result visulization
    
    ## hough transform visualization
    hough_img = bgr_img.copy();
    hough_img = hough_visualize(hough_img, lines);
    
    ## houghp transform visualization
    houghp_img = bgr_img.copy();
    for linep in linesp:
        x1,y1,x2,y2 = linep[0];
        cv2.line(houghp_img,(x1,y1),(x2,y2),(0,255,0),2);
    
    ## filter the results from houghp trans
    linesp_img = filter_linesp(linesp, bgr_img, minGrad, maxOriDiff);
    
    ## filter the results from hough trans
    lines_img = filter_lines(lines, bgr_img, minGrad, maxOriDiff);
    
    ## detect fake vanishing point(all pairs)
    (allpair_van_img, inter_points) = detect_allpair_van(lines, bgr_img);
    
    
    ## detect vanishing point()
    van_img = detect_van(inter_points, bgr_img, maxDis, minNum, maxDis_center);
    
    ## canny edge visualization
    ## write the points coordinates to file
    
    target.write(img_path+"\n");
    K = 0;
    for point in inter_points:
        target.write(str(point[:2].astype(int)));
        target.write("   ");
        if (K == 7):
            target.write("\n");
            K = 0;
        K = K+1;
    target.write("\n----------------------------------------------\n");
    ## write images to file
    cv2.imwrite("images/edges_img"+str(img_label[i])+".jpg",edges_img)
    cv2.imwrite("images/bgr_img"+str(img_label[i])+".jpg", bgr_img);
    cv2.imwrite("images/hough_img"+str(img_label[i])+".jpg", hough_img);
    cv2.imwrite("images/houghp_img"+str(img_label[i])+".jpg", houghp_img);
    cv2.imwrite("images/new_lines_img"+str(img_label[i])+".jpg", lines_img);  
    cv2.imwrite("images/allpair_van_img"+str(img_label[i])+".jpg", allpair_van_img)
    cv2.imwrite("images/van_img"+str(img_label[i])+".jpg", van_img)
print("End of the hough_trans")