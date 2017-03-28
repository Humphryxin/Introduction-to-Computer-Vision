# %load HW1.py
import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from numpy import linalg as LA

def maxabs(a, axis=None):
    """Return slice of a, keeping only those values that are furthest away
    from 0 along axis"""
    # max along specific axis
    maxa = a.max(axis=axis)
    mina = a.min(axis=axis)
    # elementwise array comparison
    p = abs(maxa) >= abs(mina) # bool, or indices where +ve values win
    n = abs(mina) > abs(maxa) # bool, or indices where -ve values win
    # deal with one value array
    if axis == None:
        if p: return maxa
        else: return mina
    out = np.zeros(maxa.shape, dtype=a.dtype)
    # Assign values based on bool array
    out[p] = maxa[p]
    out[n] = mina[n]
    return out

# read from file
img = cv2.imread('selfie.JPG')
print(img.shape)
# change the size of image
img_bgr = cv2.resize(img, (640, 480), interpolation = cv2.INTER_CUBIC);
print(img_bgr.shape)

#generate grayscale image
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY);


# Start Median filtering
med_bgr3 = cv2.medianBlur(img_bgr, 3);
med_bgr5 = cv2.medianBlur(img_bgr, 5);
med_gray3 = cv2.medianBlur(img_gray, 3);
med_gray5 = cv2.medianBlur(img_gray, 5);
# End Median Filtering


# Start Gaussian Smoothing
gaussian_bgr1 = cv2.GaussianBlur(img_bgr, (7, 7), 1);
gaussian_bgr2 = cv2.GaussianBlur(img_bgr, (13, 13), 2);
gaussian_bgr3 = cv2.GaussianBlur(img_bgr, (19, 19), 3);
gaussian_gray1 = cv2.GaussianBlur(img_gray, (7, 7), 1);
gaussian_gray2 = cv2.GaussianBlur(img_gray, (13, 13), 2);
gaussian_gray3 = cv2.GaussianBlur(img_gray, (19, 19), 3);
# End Gaussian Smoothing



# Start Image Derivatives
# kernel + correlation + magnitude + normalization
# compute kernels for computing image derivatives
kernel_x = np.array([[-1.0, 0.0, 1.0]])/2.0;
kernel_y = np.array([[-1.0], [0.0], [1.0]])/2.0;

# image derivatives in X, Y 
x_deri_gray = cv2.filter2D(img_gray, cv2.CV_32F, kernel_x)
x_median_gray3 = cv2.filter2D(med_gray3, cv2.CV_32F, kernel_x)
x_median_gray5 = cv2.filter2D(med_gray5, cv2.CV_32F, kernel_x)
x_gaussian_gray1 = cv2.filter2D(gaussian_gray1, cv2.CV_32F, kernel_x)
x_gaussian_gray2 = cv2.filter2D(gaussian_gray2, cv2.CV_32F, kernel_x)
x_gaussian_gray3 = cv2.filter2D(gaussian_gray3, cv2.CV_32F, kernel_x)

y_deri_gray = cv2.filter2D(img_gray, cv2.CV_32F, kernel_y)
y_median_gray3 = cv2.filter2D(med_gray3, cv2.CV_32F, kernel_y)
y_median_gray5 = cv2.filter2D(med_gray5, cv2.CV_32F, kernel_y)
y_gaussian_gray1 = cv2.filter2D(gaussian_gray1, cv2.CV_32F, kernel_y)
y_gaussian_gray2 = cv2.filter2D(gaussian_gray2, cv2.CV_32F, kernel_y)
y_gaussian_gray3 = cv2.filter2D(gaussian_gray3, cv2.CV_32F, kernel_y)

# test
test_gray_x = x_deri_gray.copy()
test_gray_y = y_deri_gray.copy()

# test



# Compute magnitude
mag_gray = np.sqrt(np.add(np.multiply(x_deri_gray,x_deri_gray), np.multiply(y_deri_gray, y_deri_gray)))
mag_median_gray3 = np.sqrt(np.add(np.multiply(x_median_gray3,x_median_gray3), np.multiply(y_median_gray3, y_median_gray3)))
mag_median_gray5 = np.sqrt(np.add(np.multiply(x_median_gray5,x_median_gray5), np.multiply(y_median_gray5, y_median_gray5)))
mag_gaussian_gray1 = np.sqrt(np.add(np.multiply(x_gaussian_gray1,x_gaussian_gray1), np.multiply(y_gaussian_gray1, y_gaussian_gray1)))
mag_gaussian_gray2 = np.sqrt(np.add(np.multiply(x_gaussian_gray2,x_gaussian_gray2), np.multiply(y_gaussian_gray2, y_gaussian_gray2)))
mag_gaussian_gray3 = np.sqrt(np.add(np.multiply(x_gaussian_gray3,x_gaussian_gray3), np.multiply(y_gaussian_gray3, y_gaussian_gray3)))

# Normalization
cv2.normalize(x_deri_gray,x_deri_gray,0,255, cv2.NORM_MINMAX)
x_deri_gray = np.uint8(x_deri_gray)
cv2.normalize(x_median_gray3,x_median_gray3,0,255, cv2.NORM_MINMAX)
x_median_gray3 = np.uint8(x_median_gray3)
cv2.normalize(x_median_gray5,x_median_gray5,0,255, cv2.NORM_MINMAX)
x_median_gray5 = np.uint8(x_median_gray5)
cv2.normalize(x_gaussian_gray1,x_gaussian_gray1,0,255, cv2.NORM_MINMAX)
x_gaussian_gray1 = np.uint8(x_gaussian_gray1)
cv2.normalize(x_gaussian_gray2,x_gaussian_gray2,0,255, cv2.NORM_MINMAX)
x_gaussian_gray2 = np.uint8(x_gaussian_gray2)
cv2.normalize(x_gaussian_gray3,x_gaussian_gray3,0,255, cv2.NORM_MINMAX)
x_gaussian_gray3 = np.uint8(x_gaussian_gray3)

cv2.normalize(y_deri_gray,y_deri_gray,0,255, cv2.NORM_MINMAX)
y_deri_gray = np.uint8(y_deri_gray)
cv2.normalize(y_median_gray3,y_median_gray3,0,255, cv2.NORM_MINMAX)
y_median_gray3 = np.uint8(y_median_gray3)
cv2.normalize(y_median_gray5,y_median_gray5,0,255, cv2.NORM_MINMAX)
y_median_gray5 = np.uint8(y_median_gray5)
cv2.normalize(y_gaussian_gray1,y_gaussian_gray1,0,255, cv2.NORM_MINMAX)
y_gaussian_gray1 = np.uint8(y_gaussian_gray1)
cv2.normalize(y_gaussian_gray2,y_gaussian_gray2,0,255, cv2.NORM_MINMAX)
y_gaussian_gray2 = np.uint8(y_gaussian_gray2)
cv2.normalize(y_gaussian_gray3,y_gaussian_gray3,0,255, cv2.NORM_MINMAX)
y_gaussian_gray3 = np.uint8(y_gaussian_gray3)



cv2.normalize(mag_gray,mag_gray,0,255, cv2.NORM_MINMAX)
mag_gray = np.uint8(mag_gray)
cv2.normalize(mag_median_gray3,mag_median_gray3,0,255, cv2.NORM_MINMAX)
mag_median_gray3 = np.uint8(mag_median_gray3)
cv2.normalize(mag_median_gray5,mag_median_gray5,0,255, cv2.NORM_MINMAX)
mag_median_gray5 = np.uint8(mag_median_gray5)
cv2.normalize(y_gaussian_gray1,mag_gaussian_gray1,0,255, cv2.NORM_MINMAX)
mag_gaussian_gray1 = np.uint8(mag_gaussian_gray1)
cv2.normalize(mag_gaussian_gray2,mag_gaussian_gray2,0,255, cv2.NORM_MINMAX)
mag_gaussian_gray2 = np.uint8(mag_gaussian_gray2)
cv2.normalize(mag_gaussian_gray3,mag_gaussian_gray3,0,255, cv2.NORM_MINMAX)
mag_gaussian_gray3 = np.uint8(mag_gaussian_gray3)
# End Image Derivatives



# Start Color Image Gradient
x_deri_bgr = cv2.filter2D(img_bgr, cv2.CV_32F, kernel_x)
x_deri_rgb = cv2.cvtColor(x_deri_bgr, cv2.COLOR_BGR2RGB)
y_deri_bgr = cv2.filter2D(img_bgr, cv2.CV_32F, kernel_y)
y_deri_rgb = cv2.cvtColor(y_deri_bgr, cv2.COLOR_BGR2RGB)

#
test_mag_rgb = np.sqrt(np.add(np.multiply(x_deri_rgb,x_deri_rgb), np.multiply(y_deri_rgb, y_deri_rgb)))

cv2.normalize(test_mag_rgb,test_mag_rgb,0,255, cv2.NORM_MINMAX)
test_mag_rgb = np.uint8(test_mag_rgb)

#

x_deri_rgb = maxabs(x_deri_rgb, axis=2)
y_deri_rgb = maxabs(y_deri_rgb, axis=2)
# test
test_rgb_x = x_deri_rgb.copy()
test_rgb_y = y_deri_rgb.copy()

test_diff_x = (np.absolute(test_rgb_x)-np.absolute(test_gray_x))
test_diff_y = (np.absolute(test_rgb_y)-np.absolute(test_gray_y))
print("L-infinity norm")
print("MAX =   "+str(test_diff_x.max()))
print("MIN =   "+str(test_diff_x.min()))
print("MAX =   "+str(test_diff_y.max()))
print("MIN =   "+str(test_diff_y.min()))

# test


mag_rgb = np.sqrt(np.add(np.multiply(x_deri_rgb,x_deri_rgb), np.multiply(y_deri_rgb, y_deri_rgb)))

cv2.normalize(x_deri_rgb,x_deri_rgb,0,255, cv2.NORM_MINMAX)
x_deri_rgb = np.uint8(x_deri_rgb)
cv2.normalize(y_deri_rgb,y_deri_rgb,0,255, cv2.NORM_MINMAX)
y_deri_rgb = np.uint8(y_deri_rgb)
cv2.normalize(mag_rgb,mag_rgb,0,255, cv2.NORM_MINMAX)
mag_rgb = np.uint8(mag_rgb)

# End Color Image Gradient



# Start Color Image Gradient L-2 Normalization
x_deri_bgr = cv2.filter2D(img_bgr, cv2.CV_32F, kernel_x)
x_deri_rgb = cv2.cvtColor(x_deri_bgr, cv2.COLOR_BGR2RGB)
y_deri_bgr = cv2.filter2D(img_bgr, cv2.CV_32F, kernel_y)
y_deri_rgb = cv2.cvtColor(y_deri_bgr, cv2.COLOR_BGR2RGB)

x_deri_gaussian_bgr1 = cv2.filter2D(gaussian_bgr1, cv2.CV_32F, kernel_x)
x_deri_gaussian_rgb1 = cv2.cvtColor(x_deri_gaussian_bgr1, cv2.COLOR_BGR2RGB)
y_deri_gaussian_bgr1 = cv2.filter2D(gaussian_bgr1, cv2.CV_32F, kernel_y)
y_deri_gaussian_rgb1 = cv2.cvtColor(y_deri_gaussian_bgr1, cv2.COLOR_BGR2RGB)
x_deri_gaussian_bgr2 = cv2.filter2D(gaussian_bgr2, cv2.CV_32F, kernel_x)
x_deri_gaussian_rgb2 = cv2.cvtColor(x_deri_gaussian_bgr2, cv2.COLOR_BGR2RGB)
y_deri_gaussian_bgr2 = cv2.filter2D(gaussian_bgr2, cv2.CV_32F, kernel_y)
y_deri_gaussian_rgb2 = cv2.cvtColor(y_deri_gaussian_bgr2, cv2.COLOR_BGR2RGB)
x_deri_gaussian_bgr3 = cv2.filter2D(gaussian_bgr3, cv2.CV_32F, kernel_x)
x_deri_gaussian_rgb3 = cv2.cvtColor(x_deri_gaussian_bgr3, cv2.COLOR_BGR2RGB)
y_deri_gaussian_bgr3 = cv2.filter2D(gaussian_bgr3, cv2.CV_32F, kernel_y)
y_deri_gaussian_rgb3 = cv2.cvtColor(y_deri_gaussian_bgr3, cv2.COLOR_BGR2RGB)



mag_gaussian_rgb1 = np.sqrt(np.add(np.multiply(x_deri_gaussian_rgb1,x_deri_gaussian_rgb1), np.multiply(y_deri_gaussian_rgb1, y_deri_gaussian_rgb1)))
cv2.normalize(mag_gaussian_rgb1,mag_gaussian_rgb1,0,255, cv2.NORM_MINMAX)
mag_gaussian_rgb1 = np.uint8(mag_gaussian_rgb1)
mag_gaussian_rgb2 = np.sqrt(np.add(np.multiply(x_deri_gaussian_rgb2,x_deri_gaussian_rgb2), np.multiply(y_deri_gaussian_rgb2, y_deri_gaussian_rgb2)))
cv2.normalize(mag_gaussian_rgb2,mag_gaussian_rgb2,0,255, cv2.NORM_MINMAX)
mag_gaussian_rgb2 = np.uint8(mag_gaussian_rgb2)
mag_gaussian_rgb3 = np.sqrt(np.add(np.multiply(x_deri_gaussian_rgb3,x_deri_gaussian_rgb3), np.multiply(y_deri_gaussian_rgb3, y_deri_gaussian_rgb3)))
cv2.normalize(mag_gaussian_rgb3,mag_gaussian_rgb3,0,255, cv2.NORM_MINMAX)
mag_gaussian_rgb3 = np.uint8(mag_gaussian_rgb3)

test_mag_gaussian_rgb1 = mag_gaussian_rgb1.copy()
test_mag_gaussian_rgb2 = mag_gaussian_rgb2.copy()
test_mag_gaussian_rgb3 = mag_gaussian_rgb3.copy()

# Start Norm
x_deri_gaussian_rgb1 = LA.norm(x_deri_gaussian_rgb1, axis= 2);
y_deri_gaussian_rgb1 = LA.norm(y_deri_gaussian_rgb1, axis= 2);
x_deri_gaussian_rgb2 = LA.norm(x_deri_gaussian_rgb2, axis= 2);
y_deri_gaussian_rgb2 = LA.norm(y_deri_gaussian_rgb2, axis= 2);
x_deri_gaussian_rgb3 = LA.norm(x_deri_gaussian_rgb3, axis= 2);
y_deri_gaussian_rgb3 = LA.norm(y_deri_gaussian_rgb3, axis= 2);
# End Norm

mag_gaussian_rgb1 = np.sqrt(np.add(np.multiply(x_deri_gaussian_rgb1,x_deri_gaussian_rgb1), np.multiply(y_deri_gaussian_rgb1, y_deri_gaussian_rgb1)))
cv2.normalize(mag_gaussian_rgb1,mag_gaussian_rgb1,0,255, cv2.NORM_MINMAX)
mag_gaussian_rgb1 = np.uint8(mag_gaussian_rgb1)
mag_gaussian_rgb2 = np.sqrt(np.add(np.multiply(x_deri_gaussian_rgb2,x_deri_gaussian_rgb2), np.multiply(y_deri_gaussian_rgb2, y_deri_gaussian_rgb2)))
cv2.normalize(mag_gaussian_rgb2,mag_gaussian_rgb2,0,255, cv2.NORM_MINMAX)
mag_gaussian_rgb2 = np.uint8(mag_gaussian_rgb2)
mag_gaussian_rgb3 = np.sqrt(np.add(np.multiply(x_deri_gaussian_rgb3,x_deri_gaussian_rgb3), np.multiply(y_deri_gaussian_rgb3, y_deri_gaussian_rgb3)))
cv2.normalize(mag_gaussian_rgb3,mag_gaussian_rgb3,0,255, cv2.NORM_MINMAX)
mag_gaussian_rgb3 = np.uint8(mag_gaussian_rgb3)

x_deri_rgb = LA.norm(x_deri_rgb, axis=2);
y_deri_rgb = LA.norm(y_deri_rgb, axis=2);


#test
test_rgb_x1 = x_deri_rgb.copy()
test_rgb_y1 = y_deri_rgb.copy()

test_diff_x = (np.absolute(test_rgb_x1)-np.absolute(test_rgb_x))
test_diff_y = (np.absolute(test_rgb_y1)-np.absolute(test_rgb_y))

print("L-2 norm")
print("MAX =   "+str(test_diff_x.max()))
print("MIN =   "+str(test_diff_x.min()))
print("MAX =   "+str(test_diff_y.max()))
print("MIN =   "+str(test_diff_y.min()))

#test
# End Color Image Gradient L-2 Normalization




# Start Plotting the results


#start -- plot medianblur result
fig, axarr = plt.subplots(3,2)
fig.suptitle('Median filtering')
plt.axis('off')

im = axarr[0,0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
axarr[0,0].set_title('kernel size = 0')
fig.colorbar(im, ax = axarr[0,0], orientation ='vertical')
axarr[0,0].axis('off')


im = axarr[1,0].imshow(cv2.cvtColor(med_bgr3, cv2.COLOR_BGR2RGB))
axarr[1,0].set_title('kernel size = 3')
fig.colorbar(im, ax = axarr[1,0], orientation ='vertical')
axarr[1,0].axis('off')

im = axarr[2,0].imshow(cv2.cvtColor(med_bgr5, cv2.COLOR_BGR2RGB))
axarr[2,0].set_title('kernel size = 5')
fig.colorbar(im, ax = axarr[2,0], orientation='vertical')
axarr[2,0].axis('off')

im = axarr[0,1].imshow(img_gray, cmap = 'gray')
axarr[0,1].set_title('kernel size = 0')
fig.colorbar(im, ax = axarr[0,1], orientation ='vertical')
axarr[0,1].axis('off')

im = axarr[1,1].imshow(med_gray3, cmap = 'gray')
axarr[1,1].set_title('kernel size = 3')
fig.colorbar(im, ax = axarr[1,1], orientation ='vertical')
axarr[1,1].axis('off')

im = axarr[2,1].imshow(med_gray5, cmap = 'gray')
axarr[2,1].set_title('kernel size = 5')
fig.colorbar(im, ax = axarr[2,1], orientation ='vertical')
axarr[2,1].axis('off')
plt.show()
plt.close(fig)
#end -- plot medianblur result


#start -- plot gaussianblur result
#gaussian filtering
fig, axarr = plt.subplots(4,2)
fig.suptitle('Gaussian Filtering')
plt.axis('off')


im = axarr[0,0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
axarr[0,0].set_title('Before smoothing')
fig.colorbar(im, ax = axarr[0,0], orientation ='vertical')
axarr[0,0].axis('off')


im = axarr[1,0].imshow(cv2.cvtColor(gaussian_bgr1, cv2.COLOR_BGR2RGB))
axarr[1,0].set_title('$\sigma$ = 1')
fig.colorbar(im, ax = axarr[1,0], orientation ='vertical')
axarr[1,0].axis('off')

im = axarr[2,0].imshow(cv2.cvtColor(gaussian_bgr2, cv2.COLOR_BGR2RGB))
axarr[2,0].set_title('$\sigma$ = 2')
fig.colorbar(im, ax = axarr[2,0], orientation ='vertical')
axarr[2,0].axis('off')


im = axarr[3,0].imshow(cv2.cvtColor(gaussian_bgr3, cv2.COLOR_BGR2RGB))
axarr[3,0].set_title('$\sigma$ = 3')
fig.colorbar(im, ax = axarr[3,0], orientation ='vertical')
axarr[3,0].axis('off')


im = axarr[0,1].imshow(img_gray, cmap = 'gray')
axarr[0,1].set_title('Before smoothing')
fig.colorbar(im, ax = axarr[0,1], orientation ='vertical')
axarr[0,1].axis('off')


im = axarr[1,1].imshow(gaussian_gray1, cmap = 'gray')
axarr[1,1].set_title('$\sigma$ = 1')
fig.colorbar(im, ax = axarr[1,1], orientation ='vertical')
axarr[1,1].axis('off')



im = axarr[2,1].imshow(gaussian_gray2, cmap = 'gray')
axarr[2,1].set_title('$\sigma$ = 2')
fig.colorbar(im, ax = axarr[2,1], orientation ='vertical')
axarr[2,1].axis('off')



im = axarr[3,1].imshow(gaussian_gray3, cmap = 'gray')
axarr[3,1].set_title('$\sigma$ = 3')
fig.colorbar(im, ax = axarr[3,1], orientation ='vertical')
axarr[3,1].axis('off')


plt.show()
plt.close()


#end -- plot gaussianblur result






# Start plot color image gradient


fig, axarr = plt.subplots(2,2)
plt.axis('off')
fig.suptitle('Image Gradient Gray vs Color')

im = axarr[0,0].imshow(mag_gray, cmap = 'gray')
axarr[0,0].set_title('Gray Original')
#fig.colorbar(im, ax = axarr[0,0], orientation ='vertical')
axarr[0,0].axis('off')
im = axarr[0,1].imshow(test_mag_rgb, cmap = 'gray')
axarr[0,1].set_title('Color Original')
#fig.colorbar(im, ax = axarr[0,1], orientation ='vertical')
axarr[0,1].axis('off')

im = axarr[1,0].imshow(mag_gaussian_gray1, cmap = 'gray')
axarr[1,0].set_title('Gaussian($\sigma$=1)')
#fig.colorbar(im, ax = axarr[1,0], orientation ='vertical')
axarr[1,0].axis('off')
im = axarr[1,1].imshow(test_mag_gaussian_rgb1, cmap = 'gray')
axarr[1,1].set_title('Gaussian($\sigma$=1)')
#fig.colorbar(im, ax = axarr[0,1], orientation ='vertical')
axarr[1,1].axis('off')

plt.show()
plt.close()

fig, axarr = plt.subplots(2,2)
plt.axis('off')


im = axarr[0,0].imshow(mag_gaussian_gray2, cmap = 'gray')
axarr[0,0].set_title('Gaussian($\sigma$=2)')
#fig.colorbar(im, ax = axarr[2,0], orientation ='vertical')
axarr[0,0].axis('off')
im = axarr[0,1].imshow(test_mag_gaussian_rgb2, cmap = 'gray')
axarr[0,1].set_title('Gaussian($\sigma$=2)')
#fig.colorbar(im, ax = axarr[0,1], orientation ='vertical')
axarr[0,1].axis('off')

im = axarr[1,0].imshow(mag_gaussian_gray3, cmap = 'gray')
axarr[1,0].set_title('Gaussian($\sigma$=3)')
#fig.colorbar(im, ax = axarr[3,0], orientation ='vertical')
axarr[1,0].axis('off')
im = axarr[1,1].imshow(test_mag_gaussian_rgb3, cmap = 'gray')
axarr[1,1].set_title('Gaussian($\sigma$=3)')
#fig.colorbar(im, ax = axarr[3,1], orientation ='vertical')
axarr[1,1].axis('off')

plt.show()
plt.close()

# End plot color image gradient


exit()
# Start plot image derivatives

fig, axarr = plt.subplots(3,3)
fig.suptitle('Image derivatives')
plt.axis('off')

im = axarr[0,0].imshow(x_deri_gray, cmap = 'gray')
axarr[0,0].set_title('X\noriginal', )
#fig.colorbar(im, ax = axarr[0,0], orientation ='vertical')
axarr[0,0].axis('off')

im = axarr[0,1].imshow(y_deri_gray, cmap = 'gray')
axarr[0,1].set_title('Y\noriginal')
#fig.colorbar(im, ax = axarr[0,1], orientation ='vertical')
axarr[0,1].axis('off')

im = axarr[0,2].imshow(mag_gray, cmap = 'gray')
axarr[0,2].set_title('Magnitude\noriginal')
#fig.colorbar(im, ax = axarr[0,2], orientation ='vertical')
axarr[0,2].axis('off')

im = axarr[1,0].imshow(x_median_gray3, cmap = 'gray')
axarr[1,0].set_title('Median(size=3)')
#fig.colorbar(im, ax = axarr[1,0], orientation ='vertical')
axarr[1,0].axis('off')
im = axarr[1,1].imshow(y_median_gray3, cmap = 'gray')
axarr[1,1].set_title('(size=3)')
#fig.colorbar(im, ax = axarr[1,0], orientation ='vertical')
axarr[1,1].axis('off')
im = axarr[1,2].imshow(mag_median_gray3, cmap = 'gray')
axarr[1,2].set_title('(size=3)')
#fig.colorbar(im, ax = axarr[1,0], orientation ='vertical')
axarr[1,2].axis('off')



im = axarr[2,0].imshow(x_median_gray5, cmap = 'gray')
axarr[2,0].set_title('Median(size=5)')
#fig.colorbar(im, ax = axarr[1,0], orientation ='vertical')
axarr[2,0].axis('off')
im = axarr[2,1].imshow(y_median_gray5, cmap = 'gray')
axarr[2,1].set_title('(size=5)')
#fig.colorbar(im, ax = axarr[2,0], orientation ='vertical')
axarr[2,1].axis('off')
im = axarr[2,2].imshow(mag_median_gray5, cmap = 'gray')
axarr[2,2].set_title('(size=5)')
#fig.colorbar(im, ax = axarr[1,0], orientation ='vertical')
axarr[2,2].axis('off')


#plt.show()
#plt.close()




fig, axarr = plt.subplots(3,3)
plt.axis('off')

im = axarr[0,0].imshow(x_gaussian_gray1, cmap = 'gray')
axarr[0,0].set_title('Gaussian($\sigma$=1)', )
#fig.colorbar(im, ax = axarr[0,0], orientation ='vertical')
axarr[0,0].axis('off')

im = axarr[0,1].imshow(y_gaussian_gray1, cmap = 'gray')
axarr[0,1].set_title('($\sigma$=1)')
#fig.colorbar(im, ax = axarr[0,1], orientation ='vertical')
axarr[0,1].axis('off')

im = axarr[0,2].imshow(mag_gaussian_gray1, cmap = 'gray')
axarr[0,2].set_title('($\sigma$=1)')
#fig.colorbar(im, ax = axarr[0,2], orientation ='vertical')
axarr[0,2].axis('off')

im = axarr[1,0].imshow(x_gaussian_gray2, cmap = 'gray')
axarr[1,0].set_title('Gaussian($\sigma$=2)')
#fig.colorbar(im, ax = axarr[1,0], orientation ='vertical')
axarr[1,0].axis('off')
im = axarr[1,1].imshow(y_gaussian_gray2, cmap = 'gray')
axarr[1,1].set_title('($\sigma$=2)')
#fig.colorbar(im, ax = axarr[1,0], orientation ='vertical')
axarr[1,1].axis('off')
im = axarr[1,2].imshow(mag_gaussian_gray2, cmap = 'gray')
axarr[1,2].set_title('($\sigma$=2)')
#fig.colorbar(im, ax = axarr[1,0], orientation ='vertical')
axarr[1,2].axis('off')



im = axarr[2,0].imshow(x_gaussian_gray3, cmap = 'gray')
axarr[2,0].set_title('Gaussian($\sigma$=3)')
#fig.colorbar(im, ax = axarr[1,0], orientation ='vertical')
axarr[2,0].axis('off')
im = axarr[2,1].imshow(y_gaussian_gray3, cmap = 'gray')
axarr[2,1].set_title('($\sigma$=3)')
#fig.colorbar(im, ax = axarr[2,0], orientation ='vertical')
axarr[2,1].axis('off')
im = axarr[2,2].imshow(mag_gaussian_gray3, cmap = 'gray')
axarr[2,2].set_title('($\sigma$=3)')
#fig.colorbar(im, ax = axarr[1,0], orientation ='vertical')
axarr[2,2].axis('off')



#plt.show()
#plt.close()

# END plot Derivatives



