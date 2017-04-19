(kp, des, img_sift) = def Keypoints(raw_img, feature_type)
Description: This function will take an image and compute and display keypoints/features found in the image.
Input: raw_img is the image to be processed; feature_type has two options, SIFT and SURF
Output: keypoints; descriptors; the image with keypoints plotted.

(img3, score) = def Matching(img1, img2)
Description: This function will take two images, compute the keypoints for both and display matches found.
Input: two images to be matched used SIFT features
Output: image with matches plotted; matching score which measure the distance between the two input image


() = def AllMatches(dir_path)
Description: This function should take a directory as an argument and compute pairwise matching for all images in the directory.
Display and Save: Save the distance matrix and distance image to files

() = def Hierarchical_Clustering(dis_mat)
Description: This function will cluster the images based the pairwise matrix
Display and Save: display the clusters

similarity_image.npy
Description: This file stores the distance matrix of all pairs of images in the data pool
