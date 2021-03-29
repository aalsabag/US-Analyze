import cv2
import argparse

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-v', '--video',help = "Path to the video for analysis", default = "ultrasound.mp4", type = str)
  parser.add_argument('-c', '--camera-mode', help = "Capture video stream from attached webcam", default = False, type = bool)

  arguments = parser.parse_args()

  print(arguments.video)
  #regular_search(sequence_of_interest)
  
  if arguments.camera_mode:
    vcapture = cv2.VideoCapture(0) 
    while True:
      ret, frame = vcapture.read()
      if ret == True:
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(grayscale, 75, 125)
        cv2.imshow('Edge frame', edge)
        if cv2.waitKey(20) == ord('q'):
          break
    
    vcapture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()