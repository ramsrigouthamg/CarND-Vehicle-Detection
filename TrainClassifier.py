
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
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                       visualise=False, feature_vector=feature_vec)
        return features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, orient=8,pix_per_cell=8, cell_per_block=4):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # feature_image = np.copy(image)
        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
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



if __name__ == "__main__":
    cars = glob.glob('vehicles/**/*.png', recursive=True)
    non_cars = glob.glob('non-vehicles/**/*.png', recursive=True)

    print(len(cars))
    print(len(non_cars))
    orient = 8
    pix_per_cell = 8
    cell_per_block = 1
    t1 = time.time()
    car_features = extract_features(cars,  orient=orient,
                                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)
    notcar_features = extract_features(non_cars,  orient=orient,
                                       pix_per_cell=pix_per_cell, cell_per_block=cell_per_block)
    t2 = time.time()
    print(round(t2 - t1, 3), 'No of seconds for feature extraction.')

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

    pickle.dump(svc, open("saved_svc_YCrCb.p", "wb"))
    pickle.dump(X_scaler, open("saved_X_scaler_YCrCb.p", "wb"))

    loaded_svc = pickle.load(open("saved_svc_YCrCb.p", "rb"))
    loaded_X_scale = pickle.load(open("saved_X_scaler_YCrCb.p", "rb"))

    print(loaded_svc)

    print('Test Accuracy of original SVC = ', round(svc.score(X_test, y_test), 4))
    print('Test Accuracy of pickled SVC = ', round(loaded_svc.score(X_test, y_test), 4))



