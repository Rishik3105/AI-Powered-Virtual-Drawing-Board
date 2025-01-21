import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import cv2 as cv
import mediapipe as mp
import time

class HandDetection():
    def __init__(self, mode=False, maxHands=2, detectioncon=0.5, trackcon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectioncon = float(detectioncon)
        self.trackcon = float(trackcon)
        self.mphand = mp.solutions.hands
        self.mpdraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.hand = self.mphand.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectioncon,
            min_tracking_confidence=self.trackcon
        )

    def findhands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hand.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img, handLms, self.mphand.HAND_CONNECTIONS)
        return img

    def findposition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)
        return self.lmList

    def fingersup(self):
        if not self.lmList or len(self.lmList) < max(self.tipIds):
            return []  # Return an empty list if landmarks are insufficient

        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers


def main():
    ptime = 0
    cap = cv.VideoCapture(0)
    detector = HandDetection()
    while True:
        success, img = cap.read()
        if not success:
            continue

        img = detector.findhands(img)
        lmList = detector.findposition(img,draw=False)
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv.putText(img, f'FPS: {int(fps)}', (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
        cv.imshow('Image', img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
