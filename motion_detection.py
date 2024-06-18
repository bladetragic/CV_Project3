# Brandon Bowles
# CSE4310 - 001
# Assignment 3

import sys
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QSlider
from PySide6.QtCore import Qt, Slot
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QImage, QPixmap

# Kalman Filter class
class KalmanFilter:
    def __init__(self, initial_state, initial_covariance):
        self.state = initial_state
        self.covariance = initial_covariance

    # Prediction step
    def predict(self): 
        ''' F: Transition matrix, B: Control input matrix, Q: Process noise covariance, u: Control vector'''
        F = np.eye(2) # Identity matrix for constant velocity model
        self.state = np.dot(F, self.state)
        Q = np.array([[0.1, 0], [0, 0.1]])
        self.covariance = np.dot(np.dot(F, self.covariance), F.T) + Q
    
    # Update step
    def update(self, y):
        ''' H: Measurement matrix, R: Measurement noise covariance '''
        H = np.eye(2)
        R = np.eye(2)
        S = np.dot(np.dot(H, self.covariance), H.T) + R
        K = np.dot(np.dot(self.covariance, H.T), np.linalg.inv(S))  # Kalman gain
        y = y - np.dot(H, self.state)
        self.state = self.state + np.dot(K, y)
        self.covariance = self.covariance - np.dot(np.dot(K, H), self.covariance)

# Motion Detector Class
class MotionDetector:
    def __init__(self, alpha, tau, delta, s, N):
        self.alpha = alpha
        self.tau = tau
        self.delta = delta
        self.s = s
        self.N = N

        # List to store objects being tracked
        self.objects = []  

    def update_tracking(self, frame, measurements):
        ''' Update tracking based on new frame and measurements. Add new objects if they are active over alpha frames.  
            If measurement is close to prediction of existing object, update that object '''
        
        for measurement in measurements:
            found_object_match = False
           
            for obj in self.objects:
                if np.linalg.norm(measurement - obj.kalman_filter.state) < self.delta:
                    obj.kalman_filter.update(measurement)
                    found_object_match = True
                    break
            
            if not found_object_match:
                # Create new object
                initial_state = measurement
                initial_covariance = np.eye(2)  # Identity matrix for Covariance
                new_kalman_filter = KalmanFilter(initial_state, initial_covariance)
                self.objects.append(ObjectTracker(new_kalman_filter, frame))

        # Remove inactive objects
        for obj in self.objects:
            obj.frames_since_update += 1
            if obj.frames_since_update > self.alpha:
                self.objects.remove(obj)

class ObjectTracker:
    def __init__(self, kalman_filter, frame):
        self.kalman_filter = kalman_filter
        self.prev_positions = [kalman_filter.state]
        self.frames_since_update = 0
        self.color = np.random.randint(0, 255, size=3).tolist()  # Random color for object trail

    def update_position(self, position):
        self.kalman_filter.update(position)
        self.prev_positions.append(self.kalman_filter.state)

# Media Player
class MediaPlayer(QMainWindow):
    def __init__(self, video_filename):
        super().__init__()
        self.video_filename = video_filename
        self.capture = cv2.VideoCapture(video_filename)
        self.frame_rate = self.capture.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.motion_detector = None
        self.initialize_motion_detector()
        self.initUI()

    def initialize_motion_detector(self):
        self.motion_detector = MotionDetector(alpha=1, tau=75, delta=40, s=1, N=20)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()

        # Initialize with first 3 frames
        for i in range(3):
            _, frame = self.capture.read()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.process_frame(gray_frame)

    def process_frame(self, frame):

        ''' Perform motion detection using background subtraction, Then find contours in the foreground mask '''
        foreground_mask = self.bg_subtractor.apply(frame)
        contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ''' Extract bounding boxes from contours,  Only consider large enough contours as objects '''
        measurements = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > self.motion_detector.tau:  
                # Calculate the centroid of the bounding box as the measurement
                measurement = np.array([x + w / 2, y + h / 2])
                measurements.append(measurement)

        # Update object tracking with the measurements
        self.motion_detector.update_tracking(frame, measurements)

    # def play(self):
    #     while True:
    #         ret, frame = self.capture.read()
    #         if not ret:
    #             break

    #         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         self.process_frame(gray_frame)

    #         # Display frame with tracked objects
    #         for obj in self.motion_detector.objects:
    #             color = obj.color
    #             for i in range(len(obj.prev_positions) - 1):
    #                 cv2.line(frame, tuple(obj.prev_positions[i].astype(int)), tuple(obj.prev_positions[i+1].astype(int)), color, 2)
    #             bbox = cv2.boundingRect(np.array(obj.prev_positions[-1], dtype=int))
    #             cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 2)

    #         cv2.imshow('Frame', frame)
    #         if cv2.waitKey(30) & 0xFF == ord('q'):
    #             break

    #         # Shift previous positions to maintain trails
    #         for obj in self.motion_detector.objects:
    #             obj.prev_positions.pop(0)

    #     self.capture.release()
    #     cv2.destroyAllWindows()

    def initUI(self):
        self.setWindowTitle("Object Tracking")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.label = QLabel()
        self.layout.addWidget(self.label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.total_frames - 1)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.on_slider_change)
        self.layout.addWidget(self.slider)

        self.backward_button = QPushButton("Previous Frame")
        self.backward_button.clicked.connect(self.on_backward_button_click)
        self.layout.addWidget(self.backward_button)

        self.forward_button = QPushButton("Next Frame")
        self.forward_button.clicked.connect(self.on_forward_button_click)
        self.layout.addWidget(self.forward_button)

        self.backward_60_button = QPushButton("Previous 60 Frames")
        self.backward_60_button.clicked.connect(self.on_backward_60_button_click)
        self.layout.addWidget(self.backward_60_button)

        self.forward_60_button = QPushButton("Next 60 Frames")
        self.forward_60_button.clicked.connect(self.on_forward_60_button_click)
        self.layout.addWidget(self.forward_60_button)

        self.update_frame()

    @Slot()
    def on_backward_button_click(self):
        self.current_frame = max(0, self.current_frame - 1)
        self.update_frame()

    @Slot()
    def on_forward_button_click(self):
        self.current_frame = min(self.total_frames - 1, self.current_frame + 1)
        self.update_frame()

    @Slot()
    def on_backward_60_button_click(self):
        self.current_frame = max(0, self.current_frame - 60)
        self.update_frame()

    @Slot()
    def on_forward_60_button_click(self):
        self.current_frame = min(self.total_frames - 1, self.current_frame + 60)
        self.update_frame()

    @Slot()
    def on_slider_change(self):
        self.current_frame = self.slider.value()
        self.update_frame()

    def update_frame(self):
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.capture.read()
        
        if ret:
            # Process frame
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.process_frame(gray_frame)

            # Display frame with tracked objects
            for obj in self.motion_detector.objects:
                color = obj.color
                for i in range(len(obj.prev_positions) - 1):
                    cv2.line(frame, tuple(obj.prev_positions[i].astype(int)), tuple(obj.prev_positions[i+1].astype(int)), color, 2)
                cv2.circle(frame, tuple(obj.kalman_filter.state.astype(int)), 5, color, -1)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.label.setPixmap(pixmap)
            self.slider.setValue(self.current_frame)
        else:
            self.label.setText("End of video")
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Missing video file name. Command line format should be: python program.py <video_filename>")
        sys.exit(1)

    app = QApplication(sys.argv)
    filename = sys.argv[1]
    player = MediaPlayer(filename)
    #player.play()
    player.show()
    sys.exit(app.exec())
