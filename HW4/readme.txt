(kp, des, img_sift) = def Keypoints(raw_img, feature_type)
Description: This function will take an image and compute and display keypoints/features found in the image.
Input: raw_img is the image to be processed; feature_type has two options, SIFT and SURF
Output: keypoints; descriptors; the image with keypoints plotted.

(img3, score) = def Matching(img1, img2)
Description: This function will take two images, compute the keypoints for both and display matches found.
Input: two images to be matched used SIFT features
Output: image with matches plotted; matching score which measure the distance between the two input image

(M_affine, inliers) = def findAffineTransformation(src_pts, dst_pts)
Description: This function compute the affine matrix based on ransac provided by skimage library

(dst, new_rows1, new_cols1) = def addBorder2Image(src, border_size, translation_rows, translation_cols)
Description: This function expand the image src with black pixel along the border

cropped = def cropBlackBorder(src)
Description: This function crop the black border of an image src

(dst_affine, dst_homo) = def image_stiching(img1, img2, FEATURE_OPTION, MATCHING_OPTION)
Description: This function stitches img2 to img2 using Affine and Homography for comparisons
