import cv2
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person_name": 1}
with open('labels.pickle', 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}


cap = cv2.VideoCapture(0)

while(True):

  # lấy DL từ webcam
  # ret trả về True nếu truy cập thành công
  # frame: data DL lấy được từ Webcam
  ret, frame = cap.read()
  if ret == False:
    break
  # chuyển về ảnh xám để train
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # kết hợp thư viện khuôn mặt của OpenCV
  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

  # Vẽ frame quanh khuôn mặt
  # x, y là tọa độ điểm
  # w, h kích thước khung
  for (x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]

    id_, conf = recognizer.predict(roi_gray)
    if conf >= 45:
      print(id_)
      print(labels[id_])
      font = cv2.FONT_HERSHEY_SIMPLEX
      name = labels[id_]
      color = (255, 255, 255)
      stroke = 2
      cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
    img_item = "my-image.png"
    cv2.imwrite(img_item, roi_gray)

    # vẽ hình vuông
    color = (0, 255, 0) # màu RGB - ở đây là màu xanh l
    stoke = 2
    end_core_x = x + w
    end_core_y = y + h

    cv2.rectangle(frame, (x, y), (end_core_x, end_core_y), color, stoke)
  

  # Show
  cv2.imshow('Face Detection', frame)


  # ấn q để thoát 
  if(cv2.waitKey(1) & 0xFF == ord('q')):
    break

cap.release()
cv2.destroyAllWindows()