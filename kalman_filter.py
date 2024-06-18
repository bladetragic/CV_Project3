import numpy as np

class KalmanFilter:
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        # Error check for empty matrices
        if(F is None or H is None):
            raise ValueError("Error: Please set correct system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.x = np.zeros((self.n, 1)) if x0 is None else x0 # Initial state
        self.P = np.eye(self.n) if P is None else P # Initial Error covariance matrix
        self.F = F # State transition matrix
        self.H = H # Observation matrix
        self.B = 0 if B is None else B # Control input matrix
        self.Q = np.eye(self.n) if Q is None else Q # Process noise covariance
        self.R = np.eye(self.n) if R is None else R # Measurement noise covariance
        
        # List to store previous positions
        self.history = []

    # Prediction Step
    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u) # State prediction
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q # Covariance prediction
        #self.history.append(self.x)
        return self.x

    # Update step
    def update(self, y):
        y = np.array(y).reshape((self.m, 1))
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R # Innovation covariance
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) # Kalman gain
        self.x = self.x + np.dot(K, (y - np.dot(self.H, self.x))) # State update
        I = np.eye(self.n) # Identity matrix
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)


        #self.history.append(self.x[:2])  # Store current position for visualization
        self.history.append(self.x.copy())  # Store current position for visualization

    def get_state(self):
        return self.x

    def get_history(self):
        return self.history
