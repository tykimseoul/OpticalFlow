import cv2 as cv
import numpy as np


class Sparse:
    def __init__(self, video):
        self.video = video

    def configure(self):
        # Parameters for Shi-Tomasi corner detection
        self.feature_params = dict(maxCorners=300, qualityLevel=0.2, minDistance=2, blockSize=7)
        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        # The video feed is read in as a VideoCapture object
        self.cap = cv.VideoCapture(self.video)
        # Variable for color to draw optical flow track
        self.color = (0, 255, 0)
        # ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
        self.ret, self.first_frame = self.cap.read()
        # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
        self.prev_gray = cv.cvtColor(self.first_frame, cv.COLOR_BGR2GRAY)
        # Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
        # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
        self.prev = cv.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
        # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
        self.mask = np.zeros_like(self.first_frame)

    def flow(self):
        # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
        _, frame = self.cap.read()
        # Converts each frame to grayscale - we previously only converted the first frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Calculates sparse optical flow by Lucas-Kanade method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
        next, status, _ = cv.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev, None, **self.lk_params)
        # Selects good feature points for previous position
        good_old = self.prev[status == 1]
        # Selects good feature points for next position
        good_new = next[status == 1]
        # Draws the optical flow tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            # Returns a contiguous flattened array as (x, y) coordinates for new point
            a, b = new.ravel()
            # Returns a contiguous flattened array as (x, y) coordinates for old point
            c, d = old.ravel()
            # Draws line between new and old position with green color and 2 thickness
            self.mask = cv.line(self.mask, (a, b), (c, d), self.color, 2)
            # Draws filled circle (thickness of -1) at new position with green color and radius of 3
            frame = cv.circle(frame, (a, b), 3, self.color, -1)
        # Updates previous frame
        self.prev_gray = gray.copy()
        # Updates previous good feature points
        self.prev = good_new.reshape(-1, 1, 2)
        return self.mask, frame

    def release(self):
        self.cap.release()