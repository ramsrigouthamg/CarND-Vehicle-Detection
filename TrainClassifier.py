
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.cross_validation import train_test_split


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=True, feature_vector=False)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                       visualise=False, feature_vector=feature_vec)
        return features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, orient=8,pix_per_cell=8, cell_per_block=4):
    # Create a list to append all extracted feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # An array to hold all the features (bin_spatial,color_hist and HOG)
        image_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion to 'YCrCb'
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        # Call get_hog_features() with vis=False, feature_vec=True
        spatial_features = bin_spatial(feature_image, size=(32,32))
        # Apply color_hist() also with a color space option
        hist_features = color_hist(feature_image, nbins=32)
        image_features.append(spatial_features)
        image_features.append(hist_features)
        # Call get_hog_features() with vis=False, feature_vec=True
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                 orient, pix_per_cell, cell_per_block,
                                                 vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)
        image_features.append(hog_features)
        image_features = np.concatenate(image_features)
        features.append(image_features)
    # Return list of feature vectors
    return np.array(features)

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


if __name__ == "__main__":
    # Read in all cars and non car images from the udacity dataset
    cars = glob.glob('vehicles/**/*.png', recursive=True)
    non_cars = glob.glob('non-vehicles/**/*.png', recursive=True)
    print(len(cars))
    print(len(non_cars))
    # Define hyperparameters used for HOG
    orient = 8
    pix_per_cell = 8
    cell_per_block = 1
    #  Extract the feature vectors for car and non-car images
    t1 = time.time()
    car_features = extract_features(cars,  orient=orient,
                                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)
    notcar_features = extract_features(non_cars,  orient=orient,
                                       pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)
    t2 = time.time()
    print(round(t2 - t1, 3), 'No of seconds for feature extraction.')

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    print('X.shape', X.shape)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t1 = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t1, 2), 'No of Seconds to train Support Vector Machine...')

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    print('X_train.shape', X_train.shape)

    print('X_test.shape', X_test.shape)

    import pickle

    # Save the trained support vector classifer and the scaler.
    pickle.dump(svc, open("saved_svc_YCrCb_full.p", "wb"))
    pickle.dump(X_scaler, open("saved_X_scaler_YCrCb_full.p", "wb"))

    # Load the trained support vector classifer and the scaler from disk.
    loaded_svc = pickle.load(open("saved_svc_YCrCb_full.p", "rb"))
    loaded_X_scale = pickle.load(open("saved_X_scaler_YCrCb_full.p", "rb"))

    print(loaded_svc)
    # Validate that the accuracy of original SVC and saved pickle SVC are the same.
    print('Test Accuracy of original SVC = ', round(svc.score(X_test, y_test), 4))
    print('Test Accuracy of pickled SVC = ', round(loaded_svc.score(X_test, y_test), 4))



