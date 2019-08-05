import cv2 as cv
import numpy as np


class Dense:
    def __init__(self, video):
        self.video = video

    def configure(self):
        # The video feed is read in as a VideoCapture object
        self.capture = cv.VideoCapture(self.video)
        _, self.prev_frame = self.capture.read()
        # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
        self.prev_gray = cv.cvtColor(self.prev_frame, cv.COLOR_BGR2GRAY)
        # Creates an image filled with zero intensities with the same dimensions as the frame
        self.mask = np.zeros_like(self.prev_frame)
        # Sets image saturation to maximum
        self.mask[..., 1] = 255

    def flow(self):
        # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
        _, frame = self.capture.read()
        # Opens a new window and displays the input frame
        # cv.imshow("input", frame)
        # Converts each frame to grayscale - we previously only converted the first frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Calculates dense optical flow by Farneback method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
        flow = cv.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
        # Sets image hue according to the optical flow direction
        self.mask[..., 0] = angle * 180 / np.pi / 2
        # Sets image value according to the optical flow magnitude (normalized)
        self.mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        # Converts HSV to RGB (BGR) color representation
        rgb = cv.cvtColor(self.mask, cv.COLOR_HSV2BGR)
        # Updates previous frame
        self.prev_gray = gray
        return rgb, frame

    def interpolate(self, count):
        _, frame = self.capture.read()

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 30, 15, 3, 5, 1.2, 0)
        interpolated = self.remap_image(self.prev_frame, flow, count)
        print(interpolated.shape)
        self.prev_gray = gray
        cv.imwrite("prev.jpg", self.prev_frame)
        cv.imwrite("interp.jpg", interpolated[0])
        cv.imwrite("current.jpg", frame)

    # using the flow, make new frames by interpolating from the previous frame
    def remap_image(self, prev, flow, count):
        interpolated = np.zeros((count - 1,) + prev.shape)
        for i in range(1, count):
            middle = np.copy(prev)
            for y, x in np.ndindex(prev.shape[:2]):
                xf, yf = flow[y][x]
                middle[int(round(y + yf / count * i))][int(round(x + xf / count * i))] = prev[y][x]
            interpolated[i - 1] = middle
            print(middle.shape)
        return interpolated

    def release(self):
        self.capture.release()
