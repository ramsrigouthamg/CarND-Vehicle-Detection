
import numpy as np
import cv2
import matplotlib.image as mpimg
from skimage.feature import hog
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import glob
from scipy.ndimage.measurements import label
from collections import deque

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Calculate area occupied by each bounding box
        areaInPixels = (np.max(nonzerox) - np.min(nonzerox))*(np.max(nonzeroy) - np.min(nonzeroy))
        # Draw the box on the image if it is higher than a fixed area threshold
        if areaInPixels > 5000: #In pixels
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

# Convert an image to a different color representation
def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

# Define a function to compute binned color features
def bin_spatial(image, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    color1 = cv2.resize(image[:, :, 0], size).ravel()
    color2 = cv2.resize(image[:, :, 1], size).ravel()
    color3 = cv2.resize(image[:, :, 2], size).ravel()
    # Return the feature vector
    return np.hstack((color1, color2, color3))

# Define a function to compute color histogram features
def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately
    color_1_hist = np.histogram(img[:, :, 0], bins=nbins)
    color_2_hist = np.histogram(img[:, :, 0], bins=nbins)
    color_3_hist = np.histogram(img[:, :, 0], bins=nbins)
    # Concatenate the histograms into a single feature vector and return
    return np.concatenate((color_1_hist[0], color_2_hist[0], color_3_hist[0]))

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,showImage=True):
    # Create a copy of the image
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255
    # A list to append all the bounding boxes found where detection occurred.
    bounding_boxes =[]
    # Extract the region to search from the whole image.
    img_tosearch = img[ystart:ystop, :, :]
    # Convert RGB image to YcrCb
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))
    # Split the iamge into three separate channels
    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1
    nfeat_per_block = orient * cell_per_block ** 2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    #  Iterate through the sliding windows
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            # Combine the hog features from each channel
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))
            # Extract binned color features
            spatial_features = bin_spatial(subimg, size=(32, 32))
            # Apply color_hist() also with a color space option
            hist_features = color_hist(subimg, nbins=32)
            #
            combineFeatures = np.hstack((spatial_features, hist_features, hog_features))
            # Scale features and make a prediction
            test_features = X_scaler.transform(combineFeatures).reshape(1, -1)
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                # If a car is predicted in the window append it to list of bounding boxes.
                bounding_boxes.append(((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
                if showImage:
                    cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)
    #  Return the bounding boxes and image with bounding boxes drawn
    return draw_img,bounding_boxes

# Function to call find_cars at several scales
def find_cars_multiscale(image, ystart, ystop, svc, X_scaler, orient, pix_per_cell, cell_per_block,showImage=False):
    # Different scales to search in the images for cars.
    scales = [1.0, 1.4, 1.8, 2.2, 2.6, 3.0]
    # List to collect all bounding box detections at various scales
    all_scales_bboxes = []
    # Loop through various scales
    for scale in scales:
        img,boundingboxes = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,showImage=False)
        for box in boundingboxes:
            all_scales_bboxes.append(box)
    # Return all positive detections
    return all_scales_bboxes


if __name__ == "__main__":
    # Starting and ending y coordinate to search with sliding windows
    ystart = 380
    ystop = 670
    orient = 8  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 1  # HOG cells per block
    svc = pickle.load(open("saved_svc_YCrCb_full.p", "rb"))  # Load svc
    X_scaler = pickle.load(open("saved_X_scaler_YCrCb_full.p", "rb"))  # Load svc scaler
    #  Input video to process
    cap = cv2.VideoCapture('project_video.mp4')
    # Define parameters for output video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1280, 720))
    frameNo = 0
    # Define a queue to store moving average of heat map
    heatMapMovingAverage = deque(maxlen=15)
    # Loop through the frames in the input video
    while (cap.isOpened()):
        frameNo = frameNo + 1
        print("FrameNo ", frameNo)
        # Read an image
        ret, frame = cap.read()
        # Convert bgr to rgb
        b, g, r = cv2.split(frame)
        rgb_img = cv2.merge([r, g, b])
        # Create a heatmap and an average heatmap empty array
        heat = np.zeros_like(frame[:, :, 0]).astype(np.float)
        heatAvg = np.zeros_like(frame[:, :, 0]).astype(np.float)
        # All bounding boxes with detections
        bboxes = find_cars_multiscale(rgb_img, ystart, ystop, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                      showImage=False)

        # Add heat to each box in box list
        heat = add_heat(heat, bboxes)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 5)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 1)

        # Apply threshold to help remove false positives
        heatMapMovingAverage.appendleft(heatmap)
        # Sum all the heatmaps stored in the queue.
        for eachHeatMap in heatMapMovingAverage:
            heatAvg = np.add(heatAvg,eachHeatMap)

        print ("len(heatMapMovingAverage) ",len(heatMapMovingAverage),"Max val ",np.amax(heatAvg))
        # Apply a threshold on the average heatmap.
        heatAvg = apply_threshold(heatAvg, 7)
        # Find final boxes from heatmap using label function
        labels = label(heatAvg)
        # Draw labeled boxes on the image.
        draw_img = draw_labeled_bboxes(np.copy(rgb_img), labels)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Convert bgr to rgb while writing and displaying.
        r, g, b = cv2.split(draw_img)
        final_output = cv2.merge([b, g, r])
        cv2.putText(final_output, 'Frame No: '+str(frameNo), (100, 75), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('frame', final_output)
        # Write the final image to videoWriter
        out.write(final_output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()