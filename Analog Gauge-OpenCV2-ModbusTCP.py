import numpy as np
import cv2
import time
from picamera2 import Picamera2, controls
from pyModbusTCP.server import ModbusServer
from time import sleep

# ---------------------------
# Modbus Server Setup (Slave)
# ---------------------------
server = ModbusServer(host="172.16.5.69", port=1500, no_block=True)
server.start()
print("Modbus TCP Server started on 172.16.5.69:1500")

# ---------------------------
# Calibration Inputs
# ---------------------------
print("This program will recognize gauge meter value from camera, please aim the camera to gauge!")
print("Initial parameter of gauge:")
min_angle = input('Min angle (lowest possible angle of dial) - in degrees: ')
max_angle = input('Max angle (highest possible angle) - in degrees: ')
min_value = input('Min value: ')
max_value = input('Max value: ')
units = input('Enter units: ')

# ---------------------------
# Picamera2 Setup with Continuous Autofocus
# ---------------------------
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

# Set continuous autofocus
try:
    picam2.set_controls({
        "AfMode": controls.AfModeEnum.Continuous
    })
    print("Continuous autofocus enabled.")
except Exception as e:
    print("Error enabling autofocus:", e)

# ---------------------------
# Helper Functions
# ---------------------------
def avg_circles(circles, count):
    avg_x = 0
    avg_y = 0
    avg_r = 0
    for i in range(count):
        avg_x += circles[0][i][0]
        avg_y += circles[0][i][1]
        avg_r += circles[0][i][2]
    return int(avg_x / count), int(avg_y / count), int(avg_r / count)

def dist_2_pts(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# ---------------------------
# Variables for 20-sample Averaging
# ---------------------------
sample_readings = []

# ---------------------------
# Main Processing Loop
# ---------------------------
while True:
    try:
        # Capture frame from Picamera2 (Picamera2 outputs an array in RGB)
        frame = picam2.capture_array()
        # Convert from RGB to BGR for OpenCV processing
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # --- Original Image Processing ---
        img_blur = cv2.GaussianBlur(img, (5, 5), 3)
        gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
        height, width = img.shape[:2]

        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, 
                                   np.array([]), 100, 50, 
                                   int(height * 0.35), int(height * 0.48))
        if circles is not None:
            # Convert to unsigned 16-bit and round
            circles = np.uint16(np.around(circles))
            if circles.ndim == 3:
                _, num_circles, _ = circles.shape
                x, y, r = avg_circles(circles, num_circles)
            elif circles.ndim == 2:
                x, y, r = circles
            else:
                print("Unexpected circle shape:", circles.shape)
                continue
            
            # Draw the detected circle & center
            cv2.circle(img, (x, y), r, (0,0,255), 3, cv2.LINE_AA)
            cv2.circle(img, (x, y), 2, (0,255,0), 3, cv2.LINE_AA)
        else:
            # If gauge circle isn't detected, skip
            continue

        # Draw tick marks & gauge labels
        separation = 10.0  # degrees
        interval = int(360 / separation)
        p1 = np.zeros((interval, 2))
        p2 = np.zeros((interval, 2))
        p_text = np.zeros((interval, 2))
        for i in range(interval):
            angle_rad = separation * i * np.pi / 180
            p1[i][0] = x + 0.9*r*np.cos(angle_rad)
            p1[i][1] = y + 0.9*r*np.sin(angle_rad)
            p2[i][0] = x + r*np.cos(angle_rad)
            p2[i][1] = y + r*np.sin(angle_rad)
            p_text[i][0] = x - 10 + 1.2*r*np.cos((separation*(i+9))*np.pi/180)
            p_text[i][1] = y + 5 + 1.2*r*np.sin((separation*(i+9))*np.pi/180)
        for i in range(interval):
            cv2.line(img, (int(p1[i][0]),int(p1[i][1])), (int(p2[i][0]),int(p2[i][1])), (0,255,0), 2)
            cv2.putText(img, f'{int(i*separation)}',
                        (int(p_text[i][0]), int(p_text[i][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1, cv2.LINE_AA)

        # Detect needle line
        gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, dst2 = cv2.threshold(gray2, 175, 255, cv2.THRESH_BINARY_INV)
        minLineLength = 10
        maxLineGap = 0
        lines = cv2.HoughLinesP(dst2, rho=3, theta=np.pi/180, threshold=100,
                                minLineLength=minLineLength, maxLineGap=maxLineGap)
        final_line_list = []
        diff1LowerBound = 0.15
        diff1UpperBound = 0.25
        diff2LowerBound = 0.5
        diff2UpperBound = 1.0
        for line in lines:
            for x1, y1, x2, y2 in line:
                diff1 = dist_2_pts(x, y, x1, y1)
                diff2 = dist_2_pts(x, y, x2, y2)
                if diff1 > diff2:
                    diff1, diff2 = diff2, diff1
                if (diff1 < diff1UpperBound*r and diff1 > diff1LowerBound*r and
                    diff2 < diff2UpperBound*r and diff2 > diff2LowerBound*r):
                    final_line_list.append([x1,y1,x2,y2])

        if len(final_line_list) > 0:
            x1, y1, x2, y2 = final_line_list[0]
            cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)
            
            # Determine which endpoint is the needle tip
            if dist_2_pts(x,y,x1,y1) > dist_2_pts(x,y,x2,y2):
                x_angle = x1 - x
                y_angle = y - y1
            else:
                x_angle = x2 - x
                y_angle = y - y2

            # --- Mitigating ZeroDivisionError ---
            if x_angle != 0:
                res = np.arctan(float(y_angle) / float(x_angle))
                res = np.rad2deg(res)
            else:
                # Handle the case when x_angle is zero (needle is vertical)
                if y_angle > 0:
                    res = 90  # 90 degrees if the needle is straight up
                elif y_angle < 0:
                    res = 270  # 270 degrees if the needle is straight down
                else:
                    res = 0  # 0 degrees if the needle is horizontal

            # Final angle calculation based on quadrant
            if x_angle > 0 and y_angle > 0:
                final_angle = 270 - res
            elif x_angle < 0 and y_angle > 0:
                final_angle = 90 - res
            elif x_angle < 0 and y_angle < 0:
                final_angle = 90 - res
            elif x_angle > 0 and y_angle < 0:
                final_angle = 270 - res

            # Map the angle to gauge reading
            old_min = float(min_angle)
            old_max = float(max_angle)
            new_min = float(min_value)
            new_max = float(max_value)
            old_value = final_angle
            new_value = (((old_value - old_min)*(new_max - new_min)) / (old_max - old_min)) + new_min
            int_value = int(new_value)
            print("Current reading: %d %s" % (int_value, units))
            
            # Send the reading to holding register 40000
            server.data_bank.set_holding_registers(40000, [int_value])
            print(f"Sent integer value {int_value} to register 40000")

            # Accumulate for 20-sample average & send to 40001
            sample_readings.append(int_value)
            if len(sample_readings) >= 20:
                average_value = int(sum(sample_readings) / len(sample_readings))
                server.data_bank.set_holding_registers(40001, [average_value])
                print(f"Sent average value {average_value} to register 40001")
                sample_readings = []

        # Show the final image with annotations
        cv2.imshow('Analog Gauge Reader', img)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    except (ValueError, IndexError) as e:
        print("Error processing frame:", e)
        continue

# Cleanup when 'q' is pressed
picam2.stop()
cv2.destroyAllWindows()
server.stop()
print("Program terminated.")
