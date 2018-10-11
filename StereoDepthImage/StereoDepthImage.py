
import time
import numpy as np
import cv2
import os

#cam 1 is left, cam 2 is right

def init_video():
    vc1 = cv2.VideoCapture(3)
    vc2 = cv2.VideoCapture(0)

    if vc1.isOpened() and vc2.isOpened():
        rval1, frame1 = vc1.read()
        rval2, frame2 = vc2.read()
    else:
        rval1 = False
        rval2 = False
        frame1 = None
        frame2 = None

    return vc1, vc2, rval1, rval2, frame1, frame2

def take_pictures():
    numPicsTaken = 0
    vc1, vc2, rval1, rval2, frame1, frame2 = init_video()

    while rval1 and rval2 and numPicsTaken != 30:
        rval1, frame1 = vc1.read()
        rval2, frame2 = vc2.read()

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        cv2.imshow("cam1", frame1)
        cv2.imshow("cam2", frame2)
        #--Drawing and detecting code example--

        #-Checking for chessboard in both images-
        ret1, corners1 = cv2.findChessboardCorners(gray1, (7,6), None)
        ret2, corners2 = cv2.findChessboardCorners(gray2, (7,6), None)  
        if ret1 and ret2 == True:
            numPicsTaken += 1
            img_name_a = "chessboardsA/chessboard_a{}.png".format(numPicsTaken)
            img_name_b = "chessboardsB/chessboard_b{}.png".format(numPicsTaken)
            cv2.imwrite(img_name_a, frame1)
            cv2.imwrite(img_name_b, frame2)
            print("click" + str(numPicsTaken))
            time.sleep(2)

        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
            

    cv2.destroyAllWindows()

def calibrate(imgFilePath):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for path,dirs,files in os.walk(imgFilePath):
        for filename in files:
            img = cv2.imread(imgFilePath + "/" + filename)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
            if not ret:
                print("ERROR, 1")
                print(filename)
                return False, None, None
            else:
                objpoints.append(objp)
                cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                cv2.drawChessboardCorners(gray, (7,6), corners, ret)
                #print(filename)
                #cv2.imshow('display', gray)
                #cv2.waitKey(-1)
                imgpoints.append(corners)

    return True, objpoints, imgpoints

def display():
    curDir = os.getcwd()
    objpoints = []
    imgpoints1 = []
    imgpoints2 = []

    ret1, objpoints, imgpoints1 = calibrate(curDir + "/chessboardsA")
    if ret1:
        ret2, objpoints, imgpoints2 = calibrate(curDir + "/chessboardsB")
        if ret2:
            vc1, vc2, rval1, rval2, frame1, frame2 = init_video()
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints1, gray1.shape[::-1],None,None)
            ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints2, gray2.shape[::-1],None,None)

            #v1 calib
            h1, w1 = frame1.shape[:2]
            h2, w2 = frame2.shape[:2]
            newcam1mtx, roi1 = cv2.getOptimalNewCameraMatrix(mtx1, dist1, (w1, h1), 1, (w1, h1))
            newcam2mtx, roi2 = cv2.getOptimalNewCameraMatrix(mtx2, dist2, (w2, h2), 1, (w2, h2))

            #v2 calib
            # termination criteria
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            (_, _, _, _, _, rotationMatrix, translationVector, _, _) = cv2.stereoCalibrate(
                objpoints, imgpoints1, imgpoints2,
                mtx1, dist1,
                mtx2, dist2,
                frame1.shape[:2], None, None, None, None,
                cv2.CALIB_FIX_INTRINSIC, criteria)

            (leftRectification, rightRectification, leftProjection, rightProjection,
                dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(
                        mtx1, dist1,
                        mtx2, dist2,
                        frame1.shape[:2], rotationMatrix, translationVector,
                        None, None, None, None, None,
                        cv2.CALIB_ZERO_DISPARITY, -1)

            leftMapX, leftMapY = cv2.initUndistortRectifyMap(
                    mtx1, dist1, leftRectification,
                    leftProjection, frame1.shape[:2], cv2.CV_32FC1)
            rightMapX, rightMapY = cv2.initUndistortRectifyMap(
                    mtx2, dist2, rightRectification,
                    rightProjection, frame2.shape[:2], cv2.CV_32FC1)

            stereoMatcher = cv2.StereoBM_create()
            stereo = cv2.StereoBM_create()
            #stereoMatcher.setNumDisparities(128)
            #stereoMatcher.setBlockSize(21)
            #stereoMatcher.setSpeckleRange(16)
            #stereoMatcher.setSpeckleWindowSize(45)

            num = 4

            while rval1 and rval2:
                #stereoMatcher.setMinDisparity(num)
                rval1, frame1 = vc1.read()
                rval2, frame2 = vc2.read()

                #v1 calib
                cam1v1 = cv2.undistort(frame1, mtx1, dist1, None, newcam1mtx)
                cam2v1 = cv2.undistort(frame2, mtx2, dist2, None, newcam2mtx)

                x1, y1, w1, h1 = roi1
                cam1v1 = cam1v1[y1:y1+h1, x1:x1+w1]
                x2, y2, w2, h2 = roi2
                cam2v1 = cam2v1[y2:y2+h2, x2:x2+w2]

                cv2.imshow("cam1Calibrated v1", cam1v1)
                cv2.imshow("cam2Calibrated v1", cam2v1)

                #v2 calib
                fixedLeft = cv2.remap(frame1, leftMapX, leftMapY, cv2.INTER_LINEAR)
                fixedRight = cv2.remap(frame2, rightMapX, rightMapY, cv2.INTER_LINEAR)

                grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
                grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
                depthv2 = stereoMatcher.compute(grayLeft, grayRight)

                cv2.imshow("depth v2", depthv2 / 2048)
                cv2.imshow("cam1Calibrated v2", fixedLeft)
                cv2.imshow("cam2Calibrated v2", fixedRight)

                key = cv2.waitKey(20)
                if key == 27: # exit on ESC
                    break
                elif key == ord('w'):
                    num += 1
                    print(num)
                elif key == ord('s'):
                    num -= 1
                    print(num)

if __name__ == "__main__":
    display()