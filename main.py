import cv2 as cv
from sparse_optical_flow import Sparse
from dense_optical_flow import Dense

video = "car.mp4"
sparse = Sparse(video)
dense = Dense(video)
sparse.configure()
dense.configure()


def flow():
    while sparse.cap.isOpened() and dense.capture.isOpened():
        sp, f = sparse.flow()
        dn, _ = dense.flow()
        output = cv.add(cv.add(f, sp), dn)
        # output = cv.add(f, dn)
        # Opens a new window and displays the output frame
        cv.imshow("optical flow", output)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


def interpolate():
    dense.interpolate(2)


interpolate()
sparse.release()
dense.release()
cv.destroyAllWindows()
