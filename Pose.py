import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils  # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_pose = mp.solutions.pose  # mediapipe 姿態偵測
Pose_LmsStyle = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=5)  # 關鍵點的畫圖樣式
Pose_ConStyle = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=10)  # 連接線的畫圖樣式

cap = cv2.VideoCapture(0)  # 開啟鏡頭

# 啟用姿勢偵測
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,  # 模型複雜度
    smooth_landmarks=True,  # 平滑關鍵點 減少抖動
    enable_segmentation=False,  # 去背用 需另外設置參數
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as pose:  # 相同於 pose = mp_pose.Pose()
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 將 BGR 轉換成 RGB
        results = pose.process(img2)  # 取得姿勢偵測結果
        # 根據姿勢偵測結果，標記身體節點和骨架
        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            Pose_LmsStyle,
            Pose_ConStyle,
        )
        cv2.imshow("Pose", img)
        if cv2.waitKey(5) == ord("q"):
            break  # 按下 q 鍵停止
cap.release()
cv2.destroyAllWindows()
