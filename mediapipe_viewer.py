# mediapipe_face_viewer.py

import cv2
import mediapipe as mp


# left_face_idxs = [127, 93, 113, 58, 172, 136, 150, 149, 176, 148, 152]
# right_face_idxs = [377, 400, 378, 379, 365, 397, 367, 435, 366, 447, 356]
# left_smile_idxs = [134, 131, 203, 206, 216, 212]
# right_smile_idxs = [363, 360, 423, 426, 436, 432]
# top_lip_idxs = [40, 39, 37, 0, 267, 269, 270, 409, 291, 292]
# bottom_lip_idxs = [61, 146, 91, 181, 84, 17, 314, 405, 321, 308, 324]
# middle_lip_idxs = [13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 61, 146, 91, 181, 84]
# left_eye_idxs = [33, 160, 158, 133, 153, 144, 145, 163, 7, 246]
# right_eye_idxs = [263, 362, 385, 387, 373, 380, 381, 382, 466, 388]

class FaceViewer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        self.highlighted = set()

        # Define facial regions with landmark indices
        self.regions = {
            "chin": list(range(152, 200)) + [199, 175, 152],
            "left eye": list(range(133, 144)) + [145, 153, 154, 155, 133],
            "right eye": list(range(362, 373)) + [374, 380, 381, 382, 362],
            "eyebrows": list(range(65, 72)) + list(range(55, 66)) + list(range(285, 292)) + list(range(276, 286)),
            "nose": list(range(2, 5)) + list(range(6, 10)) + list(range(94, 100)) + list(range(168, 173)) + list(range(4, 8)),
            "face trace": list(range(10)) + list(range(338, 368)) + list(range(200, 234)) + list(range(152, 200)),
        }

    def draw_landmarks(self, image, landmarks, connections):
        image_height, image_width = image.shape[:2]
        for idx, lm in enumerate(landmarks.landmark):
            x, y = int(lm.x * image_width), int(lm.y * image_height)
            color = (0, 255, 255) if idx in self.highlighted else (255, 0, 0)
            cv2.putText(image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        self.mp_drawing.draw_landmarks(
            image=image,
            landmark_list=landmarks,
            connections=connections,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        )

    def update_highlight(self, region_name):
        self.highlighted.clear()
        if region_name in self.regions:
            self.highlighted.update(self.regions[region_name])

    def run(self):
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                continue
            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    self.draw_landmarks(image, face_landmarks, self.mp_face_mesh.FACEMESH_TESSELATION)
            cv2.imshow('MediaPipe Face Viewer', image)
            key = cv2.waitKey(5)
            if key == 27:
                break
            if key == ord('\r') or key == 13:  # Enter
                cmd = input("Enter command: ").strip().lower()
                self.update_highlight(cmd)
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    viewer = FaceViewer()
    viewer.run()
