import numpy as np
import cv2
import matplotlib.image as mpimg
from skimage.feature import hog
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import glob


def process_image(img):
    global counter
    global bbox_frames

    counter += 1
    countFrame = counter % nframes

    # Find rectangles for one image
    bbox1 = find_cars_multiple(img, ystart, ystop, svc, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
                               cspace, X_scaler)

    # Find heatmap single image
    thres = smooth_thres
    heat, bboxHeat = heatmap1(img, bbox1, thres, False)
    # print('bboxHeat',bboxHeat)

    # Store the rectangles of the frame
    bbox_frames[countFrame] = bboxHeat

    # Sum rectangles of the nframes
    bbox2 = []
    for box in bbox_frames:
        if box != 0:
            for b in box:
                bbox2.append(b)
    # print('bbox2',bbox2)

    # Find heatmap of average
    thres = smooth_average
    dimg, heat, bboxHeat2 = heatmap1(img, bbox2, thres, True)

    # print('bboxHeat2',bboxHeat2)


    # Merge heatmap with image
    sizeX = int(256 * 1.3)
    sizeY = int(144 * 1.3)
    heat2 = cv2.resize(heat, (sizeX, sizeY))
    # print(np.max(heat2))
    res_img = cv2.resize(img, (sizeX, sizeY))
    res_img_gray = cv2.cvtColor(res_img, cv2.COLOR_RGB2GRAY)

    heat3 = (heat2 / np.max(heat2) * 255).astype(int)
    # print(np.max(heat3))
    # print(np.min(heat3))


    res_img_gray_R = res_img_gray  # np.zeros_like(res_img_gray)
    res_img_gray_R[(heat2 > 0)] = 255
    # img_mag_thr[(imgThres_yellow==1) | (imgThres_white==1) | (imgThr_sobelx==1)] =1
    res_img_gray_G = res_img_gray
    res_img_gray_G[(heat2 > 0)] = 0
    res_img_gray_B = res_img_gray
    res_img_gray_B[(heat2 > 0)] = 0

    # dimg[0:sizeY,0:sizeX,0]=res_img_gray_R #res_img_gray +heat3 #R
    # dimg[0:sizeY,0:sizeX,1]=res_img_gray_G
    # dimg[0:sizeY,0:sizeX,2]=res_img_gray_B

    dimg[0:sizeY, 0:sizeX, 0] = res_img_gray_R + heat3
    dimg[0:sizeY, 0:sizeX, 1] = res_img_gray
    dimg[0:sizeY, 0:sizeX, 2] = res_img_gray

    cv2.putText(dimg, "Heat Map", (102, 25), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # Highest heat value
    # for elem in lbl_heat:
    # cv2.putText(dimg,str(highv), (102,40), cv2.FONT_HERSHEY_SIMPLEX, .6,(255,0,0), 1,  lineType = cv2.LINE_AA)

    # cv2.putText(dimg,str(counter), (108,70), cv2.FONT_HERSHEY_SIMPLEX, 1.1,(255,255,255), 1,  lineType = cv2.LINE_AA)

    return dimg


if __name__ == "__main__":
    # Read in image similar to one shown above
    image = mpimg.imread('test_images/test5.jpg')

    #image = mpimg.imread('frame06.jpeg')
    #image = mpimg.imread('frame23.jpeg')
    #image = mpimg.imread('frame43.jpeg')
    #image = mpimg.imread('frame42.jpeg')

    plt.figure(figsize=(25,10))
    nframes = 1
    smooth_thres =0
    smooth_average=0
    bbox_frames=[]
    #Inicialization of list
    for i in range(nframes):
        bbox_frames.append(0)
    counter = 0

    img2 = process_image(image)
    plt.imshow(img2)
    plt.show()