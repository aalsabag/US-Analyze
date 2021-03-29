import cv2
import argparse
import random as rng
import numpy as np
rng.seed(12345)

def thresh_callback(val, src_gray):
    threshold = val
    
    canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
    
    
    contours,_ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
    
    
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    
    
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(drawing, contours_poly, i, color)
        cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
          (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
        cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
    
    
    cv2.imshow('Contours', drawing)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-v', '--video',help = "Path to the video for analysis", default = "ultrasound.mp4", type = str)
  parser.add_argument('-c', '--camera-mode', help = "Capture video stream from attached webcam", default = False, type = bool)

  arguments = parser.parse_args()

  if arguments.camera_mode:
    print("enabling camera mode")
    vcapture = cv2.VideoCapture(0) 
  elif arguments.video:
    print("enabling video capture mode")
    vcapture = cv2.VideoCapture(arguments.video) 

  while True:
    ret, frame = vcapture.read()
    if ret == True:
      src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      src_gray = cv2.blur(src_gray, (3,3))

      grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      source_window = 'Ultrasound'
      cv2.namedWindow(source_window)
      cv2.imshow(source_window, frame)

      max_thresh = 255
      thresh = 75 # initial threshold
      cv2.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, thresh_callback)
      thresh_callback(thresh,src_gray)

      edge = cv2.Canny(src_gray, 75, 125, apertureSize=3, L2gradient=True)
      cv2.imshow('Edge frame', edge)
      if cv2.waitKey(20) == ord('q'):
        break
    
  vcapture.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main()