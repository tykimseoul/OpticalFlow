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
        # ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
        _, self.prev_frame = self.cap.read()
        # self.prev_frame = cv.imread('one.png', 1)
        # self.prev_frame = cv.resize(self.prev_frame, (1920, 1200), interpolation=cv.INTER_AREA)
        # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
        self.prev_gray = cv.cvtColor(self.prev_frame, cv.COLOR_BGR2GRAY)
        # Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
        # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
        self.prev = cv.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
        # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
        self.mask = np.zeros_like(self.prev_frame)

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
            new_x, new_y = new.ravel()
            # Returns a contiguous flattened array as (x, y) coordinates for old point
            old_x, old_y = old.ravel()
            # Draws line between new and old position with green color and 2 thickness
            self.mask = cv.line(self.mask, (new_x, new_y), (old_x, old_y), (0, 255, 0), 2)
            # Draws filled circle (thickness of -1) at new position with green color and radius of 3
            frame = cv.circle(frame, (new_x, new_y), 3, (0, 255, 0), -1)
        # Updates previous frame
        self.prev_gray = gray.copy()
        # Updates previous good feature points
        self.prev = good_new.reshape(-1, 1, 2)
        return self.mask, frame

    def interpolate(self, count):
        # _, frame = self.capture.read()
        frame = cv.imread('two.png', 1)
        frame = cv.resize(frame, (1920, 1200), interpolation=cv.INTER_AREA)

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        next, status, _ = cv.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev, None, **self.lk_params)
        old_points = self.prev[status == 1]
        new_points = next[status == 1]
        for i, (new, old) in enumerate(zip(new_points, old_points)):
            new_x, new_y = new.ravel()
            old_x, old_y = old.ravel()
            self.mask = cv.line(self.mask, (new_x, new_y), (old_x, old_y), (0, 255, 0), 2)
            frame = cv.circle(frame, (new_x, new_y), 3, (0, 255, 0), -1)
        interpolated = self.remap_image(self.prev_frame, old_points, new_points, count)
        print(interpolated.shape)
        self.prev_gray = gray
        cv.imwrite("prev.jpg", cv.add(self.prev_frame, self.mask))
        cv.imwrite("interp.jpg", interpolated[0])
        # cv.imwrite("color.jpg", self.flow_to_color(flow))
        cv.imwrite("current.jpg", frame)

    def remap_image(self, prev, old_points, new_points, count):
        interpolated = np.zeros((count - 1,) + prev.shape)
        flow_amounts = new_points - old_points
        print(flow_amounts)
        for i in range(1, count):
            middle = np.zeros_like(prev)
            for y, x in np.ndindex(prev.shape[:2]):
                dists = np.sqrt((x - old_points[:, 0]) ** 2 + (y - old_points[:, 1]) ** 2)
                closest_point = np.where(dists == np.amin(dists))[0][0]
                amount = (new_points[closest_point, :] - old_points[closest_point, :])
                middle[min(int(round(y + amount[1] / count * i)), 1199)][min(int(round(x + amount[0] / count * i)), 1919)] = prev[y][x]
            interpolated[i - 1] = middle
        return interpolated

    def release(self):
        self.cap.release()
