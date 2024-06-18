# Brandon Bowles
# CSE4310 - 001
# Assignment 3

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skvideo.io
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.morphology import dilation
from scipy.spatial import distance
import kalman_filter as kf

np.float = np.float64
np.int = np.int_

class MotionDetector:
    def __init__(self, alpha, tau, delta, s, N, F, B, H, Q, R, P):
        self.alpha = alpha
        self.tau = tau
        self.delta = delta
        self.s = s
        self.N = N
        self.objects = []
        self.frame_count = 0
        self.F = F
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P

    def update(self, frame):
        self.frame_count += 1
        if self.frame_count <= 3:
            # Initialization with the first 3 frames
            self.objects.append(kf.KalmanFilter(self.F, self.B, self.H, self.Q, self.R, self.P, frame))
        elif self.frame_count % self.s == 0:
            # Skip s frames between detections
            for obj in self.objects:
                if distance.euclidean(frame, obj.get_state()) <= self.delta:
                    # Update the object's state with the new frame
                    obj.update(frame)
                else:
                    # Add a new object if it's not close to any existing object
                    if len(self.objects) < self.N:
                        self.objects.append(kf.KalmanFilter(self.F, self.B, self.H, self.Q, self.R, self.P, frame))

# class MotionDetector:
    def __init__(self, alpha=0.1, tau=10, delta=20, s=1, N=10):
        self.alpha = alpha
        self.tau = tau
        self.delta = delta
        self.s = s
        self.N = N
        self.objects = []
        self.frame = 0

    def update_tracking(self, new_frame):
        if len(self.objects) < 3:  # Initialize with first 3 frames
            self._initialize_objects(new_frame)
        else:
            if self.frame % self.s == 0:  # Update tracking every s frames
                self._detect_objects(new_frame)
                self._update_objects(new_frame)
        self.frame += 1

    def _initialize_objects(self, frame):
        # Initialize N objects with Kalman filters
        for i in range(self.N):
            # Initialize Kalman filter for each object
            initial_state = np.array([0, 0, 0, 0], dtype=float)  # [x, y, vx, vy]
            initial_covariance = np.eye(4) * 1000  # Large initial covariance
            transition_matrix = np.array([[1, 0, 1, 0],
                                           [0, 1, 0, 1],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]])
            observation_matrix = np.array([[1, 0, 0, 0],
                                           [0, 1, 0, 0]])
            control_input_matrix = np.zeros((4, 2))
            process_noise_covariance = np.eye(4) * 0.01  # Process noise covariance
            measurement_noise_covariance = np.eye(2) * 1  # Measurement noise covariance

            kalman_filter = kf.KalmanFilter(initial_state, initial_covariance, transition_matrix, observation_matrix, control_input_matrix, process_noise_covariance, measurement_noise_covariance)

            self.objects.append({'filter': kalman_filter, 'last_seen': 0})

    def _detect_objects(self, frame):
        
        # Perform motion detection and update object list
        if self.prev_frame is None:
            return  # No motion detection for the first frame

        # Convert frames to grayscale for motion detection
        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute absolute difference between frames
        frame_diff = cv2.absdiff(prev_gray, current_gray)

        # Apply threshold to filter out small changes
        _, thresholded_diff = cv2.threshold(frame_diff, self.tau, 255, cv2.THRESH_BINARY)

        # Apply morphological operations (optional)
        # thresholded_diff = cv2.morphologyEx(thresholded_diff, cv2.MORPH_OPEN, kernel)

        # Find contours in the thresholded difference image
        contours, _ = cv2.findContours(thresholded_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process each contour as a potential object candidate
        for contour in contours:
            # Compute the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)
            # Check if the contour area exceeds a minimum threshold to filter out noise
            if cv2.contourArea(contour) > self.delta:
                # Add the object candidate to the list
                self.objects.append({'bbox': (x, y, w, h), 'last_seen': 0})

    def _update_objects(self, frame):
        # Update object positions using Kalman filter prediction
        for obj in self.objects:
            obj['filter'].predict()

    def _match_objects(self, detected_objects):
    
        matched_objects = []

        for detected_obj in detected_objects:
            min_distance = float('inf')
            matched_obj = None

            for obj in self.objects:
                # Calculate the distance between the detected object and the existing object
                distance = self._calculate_distance(detected_obj, obj)

                # Check if the distance is smaller than a threshold and less than the minimum distance
                if distance < self.delta and distance < min_distance:
                    min_distance = distance
                    matched_obj = obj

            if matched_obj is not None:
                # If a matching object is found, append it to the list of matched objects
                matched_objects.append((detected_obj, matched_obj))

        return matched_objects
    
    def _calculate_distance(self, detected_obj, existing_obj):
        # Calculate the distance between the detected object and the existing object
        # This can be based on the position, bounding box overlap, or other characteristics
        # Here, let's use the Euclidean distance between the centroids of the objects' bounding boxes
        x1, y1, _, _ = detected_obj['bbox']
        x2, y2, _, _ = existing_obj['bbox']
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def _update_object_states(self, matched_objects):
        # Update state parameters of matched objects
        pass  # Implement object state updating algorithm here

    def _create_new_objects(self, unmatched_objects):
        # Create new objects for unmatched detections
        pass  # Implement object creation algorithm here

def draw_bbox(ax, bbox):
    minr, minc, maxr, maxc = bbox
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    ax.plot(bx, by, '-b', linewidth=2.5)

video_path = 'C:\\Users\\blade\\Documents\\Coding\\Python\\ComputerVision\\CV_Project3\\'
videoname = 'east_parking_reduced_size.mp4'
path2video = video_path + videoname

frames = skvideo.io.vread(path2video)

print(f'Shape of video {videoname} = {frames.shape}')

idx = 9700
threshold = 0.05


ppframe = rgb2gray(frames[idx-2]) #frame before previous frame, t-2
pframe = rgb2gray(frames[idx-1])  #previous frame, t-1
cframe = rgb2gray(frames[idx])    #current frame, t
diff1 = np.abs(cframe - pframe)
diff2 = np.abs(pframe - ppframe)

motion_frame = np.minimum(diff1, diff2)
thresh_frame = motion_frame > threshold
dilated_frame = dilation(thresh_frame, np.ones((9, 9)))
label_frame = label(dilated_frame)
regions = regionprops(label_frame)

#print(label_frame)
#print(regions)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(dilated_frame, cmap='gray')
ax.set_axis_off()
ax.margins(0, 0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

for r in regions:
    draw_bbox(ax, r.bbox)
    
plt.savefig('C:\\Users\\blade\\Documents\\Coding\\Python\\ComputerVision\\CV_Project3\\images\\objects.png', bbox_inches = 'tight',
    pad_inches = 0)

