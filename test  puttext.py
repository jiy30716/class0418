import cv2
import mediapipe as mp
import math
import numpy as np

# 初始化Mediapipe的Pose模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 初始化OpenCV影像讀取
cap = cv2.VideoCapture(0)


# 二維角度計算
def calculateAngle(landmark1, landmark2, landmark3):
    # 獲取所需座標
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3
    # 計算三點之間的夾角
    angle = math.floor(
        math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    )
    if angle < 0:
        angle += 360
    if angle > 180:
        angle = 360 - angle
    return angle


while True:
    # 讀取影像幀
    ret, image = cap.read()

    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 檢測到人體姿勢時，取得左手肘、右手肘和肩部的關鍵點座標
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
        )

        # 計算左手肘和左肩部的角度
        lms = results.pose_landmarks.landmark
        left_elbow = [
            lms[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
            lms[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
        ]
        right_elbow = [
            lms[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
            lms[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
        ]

        height, width, _ = image.shape
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            # 將關鍵點加進list內.
            landmarks.append(
                (
                    int(landmark.x * width),
                    int(landmark.y * height),
                    (landmark.z * width),
                )
            )
        l_elbow = calculateAngle(
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
        )
        r_elbow = calculateAngle(
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
        )
        l_shoulder = calculateAngle(
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
        )
        r_shoulder = calculateAngle(
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
        )
        l_hip = calculateAngle(
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
        )
        r_hip = calculateAngle(
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        )
        l_knee = calculateAngle(
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
        )
        r_knee = calculateAngle(
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
        )
        # 在左手肘的關鍵點上標記角度
        cv2.putText(
            image,
            str(l_elbow),
            (
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][0],
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][1],
            ),
            # (np.multiply(left_elbow, [image.shape[1], image.shape[0]]).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(r_elbow),
            (
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value][0],
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value][1],
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(l_shoulder),
            (
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][0],
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1],
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(r_shoulder),
            (
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0],
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1],
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(l_hip),
            (
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][0],
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][1],
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(r_hip),
            (
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][0],
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][1],
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(l_knee),
            (
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value][0],
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value][1],
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(r_knee),
            (
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value][0],
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value][1],
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # 顯示標記了角度的影像幀
    cv2.imshow("MediaPipe Pose", image)

    # 按下ESC鍵退出
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
