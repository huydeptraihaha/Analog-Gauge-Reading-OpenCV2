import numpy as np
import cv2
import time
from picamera2 import Picamera2
from libcamera import controls
from time import sleep
import math

# ---------------------------

# ---------------------------
# Calibration Inputs
# --------------------------
# ---------------------------
# Picamera2 Setup with Continuous Autofocus
# ---------------------------
def distance(x,y,cx=460,cy=440):
    return math.hypot(x-cx,y-cy)
def firstpoint(X1,Y1,X2,Y2):
    d1=distance(X1,Y1)
    d2=distance(X2,Y2)
    if d1<d2:
        return X1,Y1,X2,Y2
    else:
        return X2,Y2,X1,Y1
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (1920,1080), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

#Set continuous autofocus
try:
    picam2.set_controls({
        "AfMode": controls.AfModeEnum.Continuous
    })
    print("Continuous autofocus enabled.")
except Exception as e:
    print("Error enabling autofocus:", e)

# Main Processing Loop
# ---------------------------
while True:
    try:
        # Capture frame from Picamera2 (Picamera2 outputs an array in RGB)
        frame = picam2.capture_array()
        # Convert from RGB to BGR for OpenCV processing
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # --- Original Image Processing ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred=cv2.GaussianBlur(gray,(3,3),0)
        blurred_scan=blurred[100:980,500:1420] # kich thuoc anh dang la 920 x 880
        edges=cv2.Canny(blurred_scan,100,300) # nguong gradient tren duoi
        lines=cv2.HoughLinesP(edges,5,np.pi/180,60,minLineLength=150,maxLineGap=5)
        #rho khoang cach toi thieu
        #so vote chap nhan duong thang
        # minLineLength : chieu dai minimum 1 doan thang
        # maxLineGap khoang cach toi da giua 2 diem gan nhau de noi lai thanh doan
        # Show the final image with annotations
        if lines is not None:
            if len(lines)==1:
                x1,y1, x2,y2 = lines[0][0]
                cv2.line(blurred, (x1+500, y1+100), (x2+500,y2+100), 255, 2)
                x1,y1,x2,y2 = firstpoint(x1,y1,x2,y2)
                angle=math.degrees(math.atan2(y1-y2,x1-x2))
                print((angle+36.2766)*28.2/75)
                time.sleep(0.1)
            elif len(lines)==2:
                x1,y1, x2,y2 = lines[0][0]
                cv2.line(blurred, (x1+500, y1+100), (x2+500,y2+100), 255, 2)
                x1,y1,x2,y2 = firstpoint(x1,y1,x2,y2)
                angle1=math.degrees(math.atan2(y1-y2,x1-x2))
                x1,y1, x2,y2 = lines[1][0]
                cv2.line(blurred, (x1+500, y1+100), (x2+500,y2+100), 255, 2)
                x1,y1,x2,y2 = firstpoint(x1,y1,x2,y2)
                angle2=math.degrees(math.atan2(y1-y2,x1-x2))
                if 4<=abs(angle1-angle2)<=5:
                    angle=(angle1+angle2)/2
                    print((angle+36.2766)*28.2/75)
                    time.sleep(0.1)
                else:
                    time.sleep(0.1)
            else:
                time.sleep(0.1)
        else:
            time.sleep(0.1)
        cv2.rectangle(blurred, (500,100),(1420,980),color=0,thickness=2)
        cv2.circle(blurred,(960,540),100,color=255, thickness=-1)
        cv2.imshow('Analog Gauge Reader', blurred)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    except (ValueError, IndexError) as e:
        #
        print("Error processing frame:", e)
        continue

# Cleanup when 'q' is pressed
picam2.stop()
cv2.destroyAllWindows()
