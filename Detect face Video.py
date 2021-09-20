import cv2
from tensorflow.keras.preprocessing.image import img_to_array,load_img
import numpy as np
import CNN_predict

face = cv2.CascadeClassifier('face-detect.xml')

video = cv2.VideoCapture(0)

model = CNN_predict.Model()
counter=[0,0,0,0,0,0]

while True:
    _, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    muka = face.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in muka:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,25), 4)
        roi_warna = frame[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w] 
        Y = y-10 if y-10>10 else y+10
        cv2.imwrite('save/nama.jpg', roi_gray)
        
        img=load_img('save/nama.jpg',target_size=(200,200))
        img=img_to_array(img)
        img=np.expand_dims(img, axis=0)

        classes = ['Bill Gates','Jack Ma','Narendra Modi','Elon Musk','Mark Zuckerberg','unknown']
        conf = model.predict(img)
        idx = np.argmax(conf[0])

        if (idx==0):
                counter[0]=counter[0]+1
                counter[1]=0
                counter[2]=0
                counter[3]=0
                counter[4]=0
                counter[5]=0
                print("Verify Brangkas 1 : "+str(counter[0]))
                if (counter[0]>=3):
                    print("Brangkas 1 Terbuka")
                    
        elif(idx==1):
                counter[0]=0
                counter[1]=counter[1]+1
                counter[2]=0
                counter[3]=0
                counter[4]=0
                counter[5]=0
                print("Verify Brangkas 2 : "+str(counter[1]))
                if (counter[1]>=3):
                    print("Brangkas 2 Terbuka")

        elif (idx==2):
                counter[0]=0
                counter[1]=0
                counter[2]=counter[2]+1
                counter[3]=0
                counter[4]=0
                counter[5]=0
                print("Verify Brangkas 3 : "+str(counter[2]))
                if (counter[2]>=3):
                    print("Brangkas 3 Terbuka")

        elif (idx==3):
                counter[0]=0
                counter[1]=0
                counter[2]=0
                counter[3]=counter[3]+1
                counter[4]=0
                counter[5]=0
                print("Verify Brangkas 4 : "+str(counter[3]))
                if (counter[3]>=3):
                    print("Brangkas 4 Terbuka")

        elif (idx==4):
                counter[0]=0
                counter[1]=0
                counter[2]=0
                counter[3]=0
                counter[4]=counter[4]+1
                counter[5]=0
                print("Verify Brangkas 5 : "+str(counter[4]))
                if (counter[4]>=3):
                    print("Brangkas 5 Terbuka")

        elif (idx==5):
                counter[0]=0
                counter[1]=0
                counter[2]=0
                counter[3]=0
                counter[4]=0
                counter[5]=counter[5]+1
                print("Anda Tidak Memiliki Akses")
                if (counter[5]>=3):
                    print("Anda Tidak Memiliki Akses")
        
        label = classes[idx]
        cv2.putText(frame, label, (x, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
        
    cv2.imshow('Face', frame)
    exit = cv2.waitKey(1) & 0xff
    if exit == 27:
        break

cv2.destroyAllWindows()
video.release()
