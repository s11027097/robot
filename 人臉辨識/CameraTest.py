import numpy as np 
import cv2

cap = cv2.VideoCapture(0)
ret ,frame = cap.read()
print(cap.get(3), cap.get(4))

cv2.imshow('original frame',frame)
cap.set(3,320)
cap.set(4,240)
print(cap.get(3),cap.get(4))
while(True):
    ret,hframe = cap.read()
    cv2.imshow('half frame', hframe)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()