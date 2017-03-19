
import numpy as np
import cv2
import matplotlib.image as mpimg
from skimage.feature import hog
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import glob
from scipy.ndimage.measurements import label

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



# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(img, orient=8,pix_per_cell=8, cell_per_block=4):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images

    # Read in each one by one
    # feature_image = np.copy(image)
    feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    # Call get_hog_features() with vis=False, feature_vec=True
    hog_features = []
    for channel in range(feature_image.shape[2]):
        hog_features.append(get_hog_features(feature_image[:, :, channel],
                                             orient, pix_per_cell, cell_per_block,
                                             vis=False, feature_vec=True))
    hog_features = np.ravel(hog_features)
    features.append(hog_features)
    # Return list of feature vectors
    return features


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                   orient=8,
                   pix_per_cell=8, cell_per_block=1,
                   ):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = extract_features(test_img,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(features).reshape(1, -1)
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


if __name__ == "__main__":
    # ystart = 400
    # ystop = 656
    # orient = 8  # dist_pickle["orient"]
    # pix_per_cell = 8  # dist_pickle["pix_per_cell"]
    # cell_per_block = 1  # dist_pickle["cell_per_block"]
    # spatial_size = (32, 32)  # dist_pickle["spatial_size"]
    # hist_bins = 32  # dist_pickle["hist_bins"]
    # cspace = 'YCrCb'

    ### TODO: Tweak these parameters and see how the results change.
    color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 8  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 1  # HOG cells per block
    y_start_stop = [400, 656]  # Min and max in y to search in slide_window()
    svc = pickle.load(open("saved_svc_YCrCb.p", "rb"))  # Load svc
    X_scaler = pickle.load(open("saved_X_scaler_YCrCb.p", "rb"))  # Load svc


    # Read in image similar to one shown above
    image = mpimg.imread('test_images/test6.jpg')
    draw_image = np.copy(image)

    # image = image.astype(np.float32) / 255

    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                           xy_window=(64, 64), xy_overlap=(0.75, 0.75))

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block)

    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    # window_img = draw_boxes(draw_image, windows, color=(0, 0, 255), thick=6)
    plt.imshow(window_img)
    plt.show()