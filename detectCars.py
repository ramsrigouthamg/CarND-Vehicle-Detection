import numpy as np
import cv2
import matplotlib.image as mpimg
from skimage.feature import hog
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import glob
from scipy.ndimage.measurements import label
# from moviepy.editor import VideoFileClip
# import moviepy as mve

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(labels, heatmap, img=None):
    # Iterate through all detected cars
    b_heat = []  # ((x1,y1),(x2,y2))

    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        highv = np.sum(heatmap[np.min(nonzeroy):np.max(nonzeroy), np.min(nonzerox):np.max(nonzerox)])

        # Draw the box on the image
        if not img is None:
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 200), 3)

        b_heat.append(((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))))

    # Return the image
    if not img is None:
        return img, b_heat
    else:
        return b_heat


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features

def heatmap1(image, box_list, threshold=1, showImg=True):
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, box_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, threshold)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    if showImg == True:
        draw_img, lbl_heat = draw_labeled_bboxes(labels, heatmap, np.copy(image))
        return draw_img, heatmap, lbl_heat
    else:
        lbl_heat = draw_labeled_bboxes(labels, heatmap)
        return heatmap, lbl_heat

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, cspace,
              X_scaler, showImage=True):
    count = 0
    draw_img = np.copy(img)
    # img = img.astype(np.float32)/255
    boxes = []

    img_tosearch = img[ystart:ystop, :, :]
    # print(ystart,ystop)
    # print('img', img.shape)
    # print('img_tosearch', img_tosearch.shape)

    # ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if cspace != 'RGB':
        if cspace == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else:
        ctrans_tosearch = np.copy(img_tosearch)

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1
    nfeat_per_block = orient * cell_per_block ** 2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    # nblocks_per_window = (window // pix_per_cell)-1
    nblocks_per_window = (window // pix_per_cell)

    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            # print(yb ,' ypos ',ypos)
            # print(yb ,' ypos ',ypos+nblocks_per_window)

            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].reshape(1, -1)
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].reshape(1, -1)
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].reshape(1, -1)
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            # subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features
            # spatial_features = bin_spatial(subimg, size=spatial_size)
            # hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            # test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # print('hog_features.shape',hog_features.shape)

            # X_scaler = StandardScaler().fit(hog_features)
            test_features = X_scaler.transform(hog_features).reshape(1, -1)

            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                c1 = np.random.randint(0, 255)
                c2 = np.random.randint(0, 255)
                c3 = np.random.randint(0, 255)
                count += 1
                if showImage == True:
                    cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                                  (xbox_left + win_draw, ytop_draw + win_draw + ystart), (c1, c2, c3), 6)
                    cv2.putText(draw_img, str(count), (int(xbox_left), int(ytop_draw + ystart)),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, lineType=cv2.LINE_AA)
                else:
                    if count > 0:
                        boxes.append(
                            ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))

    if showImage:
        return draw_img, count
    else:
        return boxes, count


# Run find_cars in several scales
def find_cars_multiple(img, ystart, ystop, svc, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, cspace,
                       X_scaler):
    # Bom scales = [1.1,1.4, 1.8, 2.2, 2.7, 3.2]
    scales = [1.1, 1.4, 1.8, 2.4, 2.9, 3.4]

    c = 0
    bbox = []
    for scale in scales:
        c += 1
        # The first half of scales is valid for the upper half of image
        if c < len(scales):
            ystartaux = ystart
            ystopaux = int((ystart + ystop) / 2)
        else:
            ystartaux = int((ystart + ystop) / 2)
            ystopaux = ystop

        box, count = find_cars(img, ystartaux, ystopaux, scale, svc, orient, pix_per_cell, cell_per_block, spatial_size,
                               hist_bins, cspace, X_scaler, False)
        if count > 0:
            for b in box:
                bbox.append(b)
    return bbox

ystart = 400
ystop = 656
orient = 8 #dist_pickle["orient"]
pix_per_cell = 8 #dist_pickle["pix_per_cell"]
cell_per_block = 1 # dist_pickle["cell_per_block"]
spatial_size = (32, 32) #dist_pickle["spatial_size"]
hist_bins = 32 # dist_pickle["hist_bins"]
cspace = 'YCrCb'
svc = pickle.load( open("saved_svc_YCrCb.p", "rb" ) ) #Load svc
X_scaler = pickle.load( open("saved_X_scaler_YCrCb.p", "rb" ) ) #Load svc

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
    # image = mpimg.imread('test_images/test3.jpg')
    #
    # nframes = 1
    # smooth_thres =0
    # smooth_average=0
    # bbox_frames=[]
    # #Inicialization of list
    # for i in range(nframes):
    #     bbox_frames.append(0)
    # counter = 0
    #
    # img2 = process_image(image)
    # plt.imshow(img2)
    # plt.show()

    # Create video file pipeline
    counter = 0
    nframes = 25
    smooth_thres = 1
    smooth_average = 6

    bbox_frames = []
    # Inicialization of list
    for i in range(nframes):
        bbox_frames.append(0)
    counter = 0


    cap = cv2.VideoCapture('project_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1280, 720))
    frameNo = 0
    while (cap.isOpened()):
        frameNo = frameNo + 1
        print ("FrameNo ",frameNo)
        # Read an image
        ret, frame = cap.read()
        b,g,r = cv2.split(frame)
        rgb_img = cv2.merge([r,g,b])
        final_output =process_image(rgb_img)

        r, g, b = cv2.split(final_output)
        final_output = cv2.merge([b,g,r])
        cv2.imshow('frame', final_output)
        # Write the final image to videoWriter
        out.write(final_output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # clip1 = VideoFileClip("project_video.mp4")
    # clip1 = VideoFileClip("project_video.mp4")  # .subclip(40,44)
    #
    # out_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    # out_clip.write_videofile(output, audio=False)