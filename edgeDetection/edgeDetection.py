#
# Background exclusion using Edge Detection
#
# How to use:
#     put image file at './clothes', 
#     then get background excluded result from './result'
#
# NOTE: 
#     1) Each image takes 5~10 sec. (when size ~ 1920x1280)
#     2) this does not work well with images with low contrast of clothes and background. 
#        threshold values can be optimized for better performance. 
#     3) This program assumes clothes are not twisted and has no closed-loop background
#

import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

datadir = './clothes'
resultdir = './result'
showIntermediateResults = True

def detect(path):
    # Read the original image
    img = cv2.imread(path)

    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 

    # Canny Edge Detection
    # how do we decide threshold values??
    edges = cv2.Canny(image=img_blur, threshold1=50, threshold2=100)

    # Generate image mask
    npEdges = np.array(edges)
    M, N = npEdges.shape
    mask = np.zeros((M, N), dtype=np.uint8)
    mask[:, :] = 255
    for i in range(M):
        j = 0
        while j < N and npEdges[i, j] == 0:
            j += 1
        start = j
        j = N
        while j > 0 and npEdges[i, j-1] == 0:
            j -= 1
        end = j
        mask[i, 0:start] = 0
        mask[i, end:N] = 0
    for j in range(N):
        i = 0
        while i < M and npEdges[i, j] == 0:
            i += 1
        start = i
        i = M
        while i > 0 and npEdges[i-1, j] == 0:
            i -= 1
        end = i
        mask[0:start, j] = 0
        mask[end:M, j] = 0

    # Leave clothes only (result)
    maskedImg = np.ndarray((M, N, 4), dtype=np.uint8)
    image = np.array(img)
    maskedImg[:, :, :3] = image[:, :, :]
    maskedImg[:, :, 3] = mask

    # Save the result
    cv2.imwrite(resultdir+'/'+path.split('/')[2].split('.')[-2]+'.png', maskedImg)

    # Show the result
    if showIntermediateResults:
        plt.figure(figsize=(12,8))
        plt.subplot(221), plt.imshow(img)
        plt.title("Original image"), plt.axis('off')
        plt.subplot(222), plt.imshow(npEdges, cmap='gray')
        plt.title("Edge Detection"), plt.axis('off')
        plt.subplot(223), plt.imshow(mask, cmap='gray')
        plt.title("Image mask"), plt.axis('off')
        plt.subplot(224), plt.imshow(maskedImg)
        plt.title("Result"), plt.axis('off')
        plt.show()

def main():
    for img_path in glob.glob(datadir+'/*.*'):
        detect(img_path)

if __name__ == '__main__':
    main()