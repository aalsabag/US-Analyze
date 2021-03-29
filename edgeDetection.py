import cv2
import argparse

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
      grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      edge = cv2.Canny(grayscale, 75, 125, apertureSize=3, L2gradient=True)
      cv2.imshow('Edge frame', edge)
      if cv2.waitKey(20) == ord('q'):
        break
    
  vcapture.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main()