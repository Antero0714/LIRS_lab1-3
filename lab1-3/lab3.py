import cv2
import time

class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_eye.xml'
        )
        self.smile_cascade = cv2.CascadeClassifier(
            cascade_path + 'haarcascade_smile.xml'
        )
        self.prev_time = time.time()
        self.fps = 0

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5,
            minSize=(30, 30), maxSize=(300, 300)
        )
        
        all_eyes = []
        all_smiles = []
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
            smiles = self.smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
            all_eyes.append(eyes)
            all_smiles.append(smiles)
        
        return faces, all_eyes, all_smiles

    def draw(self, frame, faces, all_eyes, all_smiles):
        frame_copy = frame.copy()
        has_smile = False
        eyes_open = True
        
        for idx, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            eyes = all_eyes[idx]
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame_copy, (x+ex, y+ey), 
                            (x+ex+ew, y+ey+eh), (255, 0, 0), 2)
            
            if len(eyes) < 2:
                eyes_open = False
            
            smiles = all_smiles[idx]
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(frame_copy, (x+sx, y+sy), 
                            (x+sx+sw, y+sy+sh), (0, 0, 255), 2)
                has_smile = True
        
        return frame_copy, has_smile, eyes_open

    def update_fps(self):
        current_time = time.time()
        if current_time - self.prev_time > 0:
            self.fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time

    def run(self):
        print("ЛР №3: Детекция лиц (нажмите 'q' для выхода)")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Ошибка: камера не открывается")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        fps_list = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.update_fps()
            fps_list.append(self.fps)
            
            faces, eyes, smiles = self.detect(frame)
            frame_result, has_smile, eyes_open = self.draw(frame, faces, eyes, smiles)
            
            if not has_smile:
                cv2.putText(frame_result, "Smile!", (50, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            
            if not eyes_open:
                cv2.putText(frame_result, "Otkroy glaza!", (50, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            
            cv2.putText(frame_result, f"FPS: {self.fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Face Detection', frame_result)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if fps_list:
            print(f"Средний FPS: {sum(fps_list)/len(fps_list):.2f}")
        print("Готово!")

if __name__ == "__main__":
    detector = FaceDetector()
    detector.run()
