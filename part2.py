import cv2             # OpenCV套件
import mediapipe as mp  # Mediapipe套件

mp_pose = mp.solutions.pose  # mediapipe 姿態偵測

cap = cv2.VideoCapture("Jumping Jacks.mp4")  # 開啟鏡頭

# 啟用姿勢偵測
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,  # 模型複雜度
    smooth_landmarks=True,  # 平滑關鍵點 減少抖動
    enable_segmentation=False,  # 去背用 需另外設置參數
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as pose:  # 相同於 pose = mp_pose.Pose()
    
    while True:
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 將 BGR 轉換成 RGB
        results = pose.process(img2)  # 取得姿勢偵測結果
        # 根據姿勢偵測結果，標記身體節點和骨架
        if results.pose_landmarks:
            print(results.pose_landmarks)
            break
        cv2.imshow("Pose", img)
        if cv2.waitKey(5) == ord("q"):
            break  # 按下 q 鍵停止
cap.release()
cv2.destroyAllWindows()
