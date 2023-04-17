import cv2             # OpenCV套件
import mediapipe as mp  # Mediapipe套件

cap = cv2.VideoCapture("Jumping Jacks.mp4")  # 開啟鏡頭

while True:
    ret, img = cap.read()  # 鏡頭讀出的布林值跟Frame
    if not ret:
        print("Cannot receive frame")
        break
  
    cv2.imshow("Pose", img)
    if cv2.waitKey(15) == ord("q"):
        break  # 按下 q 鍵停止
cap.release()
cv2.destroyAllWindows()
