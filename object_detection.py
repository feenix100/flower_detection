import torch
import cv2
import numpy as np

#this script will use detect objects using a custom yolov5 image detection model
#it will detect objects in a video or live video or images 

# Load the custom YOLOv5 model
model_path = 'path/to/your/custom_yolov5_model.pt'  # Update this path
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

def detect_objects_in_image(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return

    # Convert the image from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(img_rgb)

    # Results
    results.print()  # Print results to console
    results.show()  # Show results

    # Extract bounding boxes and labels
    boxes = results.xyxy[0].cpu().numpy()  # Bounding boxes
    labels = results.names  # Class labels

    # Draw bounding boxes on the image
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        label = labels[int(cls)]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f'{label} {conf:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save the result image
    result_image_path = 'path/to/save/detected_image.jpg'  # Update this path
    cv2.imwrite(result_image_path, img)
    print(f"Detection result saved to {result_image_path}")

def detect_objects_in_video(video_source=0):
    # Open video source (default is the first connected camera)
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform inference
        results = model(frame_rgb)

        # Extract bounding boxes and labels
        boxes = results.xyxy[0].cpu().numpy()  # Bounding boxes
        labels = results.names  # Class labels

        # Draw bounding boxes on the frame
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            label = labels[int(cls)]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame with bounding boxes
        cv2.imshow('YOLOv5 Object Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close display window
    cap.release()
    cv2.destroyAllWindows()

# Example usage
image_path = 'path/to/your/image.jpg'  # Update this path
detect_objects_in_image(image_path)

# For live video detection, call the function without arguments for the default camera
# or provide a video file path or camera index as argument.
detect_objects_in_video()
