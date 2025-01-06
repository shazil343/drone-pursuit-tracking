import cv2
import numpy as np
import random
import math

# Simulated Lidar data generator
def simulate_lidar_data():
    """
    Simulates Lidar data by generating random angle and distance values.
    """
    angle = random.uniform(0, 360)  # Angle in degrees
    distance = random.uniform(500, 2000)  # Distance in mm (adjust range as needed)
    return angle, distance

# Convert polar coordinates (angle, distance) to Cartesian (x, y, z)
def polar_to_cartesian(angle_deg, distance_mm):
    angle_rad = math.radians(angle_deg)
    x = distance_mm * math.cos(angle_rad)
    y = distance_mm * math.sin(angle_rad)
    z = distance_mm  # For simplicity, assume z is the distance
    return x, y, z

# Load YOLOv3-Tiny model with correct weights and configuration files
net = cv2.dnn.readNet("/Users/user/Desktop/Drone Surveillance Python/yolov3-tiny.weights", 
                      "/Users/user/Desktop/Drone Surveillance Python/yolov3-tiny.cfg")

# Get the names of the output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load COCO labels
with open("/Users/user/Desktop/Drone Surveillance Python/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture from phone's IP camera
cap = cv2.VideoCapture("http://10.14.118.222:8080/video")  # Replace with your phone's IP

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Process every frame for YOLO
    if frame_count % 1 == 0:
        # Resize input to (416, 416) for improved detection accuracy
        blob = cv2.dnn.blobFromImage(frame, 1/255, (156, 156), (104, 117, 123), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layers)

        # Get frame dimensions
        height, width, channels = frame.shape

        # Process YOLO detections
        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.1:
                    # Calculate bounding box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Simulate Lidar data
                    angle, distance = simulate_lidar_data()
                    x3d, y3d, z3d = polar_to_cartesian(angle, distance)
                    print(f"3D Position: x={x3d:.2f}, y={y3d:.2f}, z={z3d:.2f}")

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f"{classes[class_id]}: {confidence:.2f}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("YOLOv3-Tiny Object Detection with Simulated Lidar", frame)

    # Increment frame count
    frame_count += 1

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

