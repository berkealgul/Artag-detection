import cv2
import numpy as np
import json


def save_config(mtx, dist, filename='default'):
    data = {}
    data['mtx'] = mtx.tolist()
    data['dist'] = dist.tolist()
    with open(filename + '.json', 'w') as saveFile:
        json.dump(data, saveFile)


def obtain_coordinate(frame, boardSize, showResult=False):
    chessBoardFlag = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    found, corners = cv2.findChessboardCorners(frame, boardSize, flags=chessBoardFlag)

    if found is True:
        corners2 = cv2.cornerSubPix(frame, corners, (5, 5), (-1, -1), criteria)

        if showResult is True:
            cv2.drawChessboardCorners(frame, boardSize, corners2, found)
            cv2.imshow("chessboard", frame)

        return corners2
    else:
        return None


def calibrate_camera(img, objPoints, imgPoints):
    ret, mtx, dist, rvecs, tvec = cv2.calibrateCamera(objPoints, imgPoints, img.shape[::-1], None, None)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    return newcameramtx, mtx, dist


def prepare_coordinates(imageCount, boardSize, objp, cam):
    done = True

    imgPoints = list()
    objPoints = list()

    i = 0
    while i < imageCount:
        _, img = cam.read()
        cv2.imshow('frame', img)

        key = cv2.waitKey(20)
        if key == 27:  # esc ile çıkar
            done = False
            break
        elif key != 97:  # küçük a ile işlem yaparız
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgP = obtain_coordinate(gray, boardSize)

        if imgP is None:
            print("Desen bulunamadı açıyı düzeltiniz")
            continue

        imgPoints.append(imgP)
        objPoints.append(objp)
        i += 1

        print("Kalibrasyon = %" + str(int(i * 100 / imageCount)))

    return imgPoints, objPoints, done


# NOT esc ile çıkarsın a tuşu ile resim kaydedersin
# ikinci aşamda a tuşu ile kalibrasyo ndosyasını kaydedersin esc ile çıkarsın

# parametre tanıllama
boardSquareDim = 0.02  # meters
imageCount = 50
boardSize = (7, 9)
saveFileName = 'logi-g922-config'

objp = np.zeros((boardSize[0]*boardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:boardSize[0], 0:boardSize[1]].T.reshape(-1, 2)

cam = cv2.VideoCapture(1)
_,img = cam.read()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# veriler kullanıcıdan alınır
imgPoints, objPoints, done = prepare_coordinates(imageCount, boardSize, objp, cam)

newcameramtx = None
mtx = None
dist = None

if done is True:
    # asıl iş burada yapılır
    newcameramtx, mtx, dist = calibrate_camera(img, objPoints, imgPoints)

while done is True:
    _,img = cam.read()

    key = cv2.waitKey(20)
    if key == 27:
        break
    elif key == 97: # a tuşu ile kaydederiz
        save_config(mtx, dist, filename=saveFileName)
        break

    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    cv2.imshow("undistorted", dst)
    cv2.imshow("frame", img)

cv2.destroyAllWindows()