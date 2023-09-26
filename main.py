import cv2
import numpy as np
import argparse
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.05)

# создаем пустую функцию
def nothing(args): pass


# создаем окно для отображения результата и бегунки
cv2.namedWindow("setup")
cv2.createTrackbar("h1", "setup", 0, 255, nothing)
cv2.createTrackbar("v1", "setup", 0, 255, nothing)
cv2.createTrackbar("s1", "setup", 0, 255, nothing)
cv2.createTrackbar("h2", "setup", 249, 350, nothing)
cv2.createTrackbar("v2", "setup", 249, 350, nothing)
cv2.createTrackbar("s2", "setup", 249, 350, nothing)
cv2.createTrackbar("ex", "setup", 0, 51, nothing)

while True:
    _, frame = cap.read()

    height, width = frame.shape[0:2]  # получаем разрешение кадра
    h1 = cv2.getTrackbarPos('h1', 'setup')
    s1 = cv2.getTrackbarPos('s1', 'setup')
    v1 = cv2.getTrackbarPos('v1', 'setup')
    h2 = cv2.getTrackbarPos('h2', 'setup')
    s2 = cv2.getTrackbarPos('s2', 'setup')
    v2 = cv2.getTrackbarPos('v2', 'setup')
    ex = cv2.getTrackbarPos('ex', 'setup')

    # собираем значения из бегунков в множества
    min_p = (h1, s1, v1)
    max_p = (h2, s2, v2)

    hvs_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #hvs_img = cv2.GaussianBlur(hvs_img, (3, 3), 20)

    frame_threshold = cv2.inRange(hvs_img, (min_p), (max_p))
    contours, _ = cv2.findContours(frame_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) !=0:
        maxc = max(contours, key=cv2.contourArea)
        moments = cv2.moments(maxc)

        if moments["m00"] > 20:
            cx = int(moments["m10"]/moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            iSee = True  # устанавливаем флаг, что контур найден
            controlX = 2 * (cx - width / 2) / width  # находим отклонение найденного объекта от центра кадра и нормализуем его (приводим к диапазону [-1; 1])
            cv2.drawContours(frame, maxc, -1, (0, 255, 0), 2)  # рисуем контур
            cv2.line(frame, (cx, 0), (cx, height), (0, 255, 0), 2)  # рисуем линию линию по x
            cv2.line(frame, (0, cy), (width, cy), (0, 255, 0), 2)  # линия по y

    cv2.putText(frame, 'iSee: {};'.format(iSee), (width - 370, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)  # текст
    cv2.putText(frame, 'controlX: {:.2f}'.format(controlX), (width - 200, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)  #
    cv2.imshow("frame", frame)
    cv2.imshow("frame_threshold", frame_threshold)
    cv2.imshow("HVS", hvs_img)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
