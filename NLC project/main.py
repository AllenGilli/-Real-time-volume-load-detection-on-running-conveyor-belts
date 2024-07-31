import cv2
import numpy as np
import csv
import time
import math
import os
from datetime import datetime

#input video
cap = cv2.VideoCapture('conveyor_belt.mp4')

# Conversion factor from pixels to centimeters
pixels_to_cm = 171 / 1040
#base length of conveyor
base = 53
#inclined angle of the conveyor
angle=45
angle=math.radians(angle)
#density of load
density=0.9 #lignite density

# Output CSV files
width_csv = 'width_values.csv'
volume_csv = 'average_volume.csv'
total_volume_csv = 'total_volume_per_shift.csv'
csv_files = [width_csv, volume_csv,total_volume_csv]

for file in csv_files:
    if os.path.exists(file):
        os.remove(file)

width_values = []
volume_values=[]
start_time = time.time()
current_time=0
start_time_t=time.time()
interval_duration = 5  # Width calculation interval in +seconds
speed=4.2 # Speed in m/s

def volume_calc(volume_val,start_time_t,current_time):
    total_volume = sum(volume_val)
    total_volume = total_volume/1000000
    total_weigth = total_volume*density
    # Store the total volume in the new CSV file with date and shift
    shift_start_time = datetime.fromtimestamp(start_time_t).strftime('%Y-%m-%d %H:%M:%S')
    shift_end_time = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')
    shift_info = f"{shift_start_time} to {shift_end_time}"

    with open(total_volume_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([shift_info, total_weigth])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame=cv2.resize(frame,(680,420))
    # Obtain frame dimensions
    height, width, _ = frame.shape

    # Define the region of interest (ROI) in the middle of the frame
    scaling_factor = 0.5 # Adjust this factor to make the ROI bigger or smaller
    roi_width = int(width * scaling_factor)+100 # Width of the ROI
    roi_height = int(height * scaling_factor)  # Height of the ROI
    roi_x = int((width - roi_width) / 2)  # X coordinate of the ROI (centered)
    roi_y = int((height - roi_height) / 2) # Y coordinate of the ROI (centered)

    # Create the ROI by extracting the middle portion of the frame
    roi = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

    # Convert the ROI to grayscale and perform object detection within this region
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    x,y,w,h=0,0,0,0
    width_in_cm = 0
    # Combine contours in the middle by creating a bounding rectangle around all detected contours
    if len(contours) > 0:
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))

        # Convert width from pixels to centimeters
        width_in_cm = w * pixels_to_cm

        # Store width value in the CSV file
        with open(width_csv, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([time.time(), width_in_cm])

        width_values.append(width_in_cm)

    # Check if interval_duration has passed
    current_time = time.time()
    if current_time - start_time >= interval_duration:
        # Calculate average width
        if width_values:
            average_width = sum(width_values) / len(width_values)
            # Delete widths from the CSV file
            with open(width_csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # write empty to clear the file
                writer.writerow([])
            height = (abs(average_width - 53) / 2) * math.tan(angle)
            volume = ((53 + average_width) * height) / 2
            volume = speed * 5 * volume * 100
            volume_values.append(volume)

            # Store the average volume in the average CSV file with time (IST)
            ist_time = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')
            with open(volume_csv, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([ist_time, volume])

            # Clear the width values for the next interval
            width_values = []

            # Update the start time for the next interval
            start_time = current_time

        #check if 20 second(or shift interval) has passed
        if current_time - start_time_t >=20:
            # Calculate total volume for the last 20 seconds
            volume_calc(volume_values,start_time_t,current_time)

            # Clear the volume values for the next interval
            volume_values = []
            start_time_t=current_time

    # Draw a single bounding rectangle around all the detected contours in the middle
    cv2.rectangle(roi, (x, y), (x + w, y + h), color=(255, 255, 0), thickness=2)
    cv2.putText(roi, f'Width: {width_in_cm:.2f} cm', (x+5, y +30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                (255, 0, 0), 1)
    # Display the modified frame with a single bounding rectangle around detected objects in the middle
    cv2.imshow("Output", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):  # Adjust the parameter for your preferred frame rate
        break

volume_calc(volume_values,start_time_t,current_time)
# Release resources
cap.release()
cv2.destroyAllWindows()