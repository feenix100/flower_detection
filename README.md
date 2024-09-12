# Custom Image Detection Models for Cannabis Buds

### Overview

This repository contains custom image detection models trained with YOLO v5, designed specifically for detecting cannabis buds (flowers).

### Features

- **Detection Capability**: These models can detect cannabis buds in both images and live video.
- **Bounding Box**: Once a detection occurs, a bounding box is drawn around the detected flower.
- **OAK-D Pro Integration**: Scripts are available to utilize the YOLO v5 custom models with the OAK-D Pro camera.

### Usage Instructions

1. **Download Models**: Download one of the custom YOLO v5 models.
2. **Integrate into Python Script**: Add the downloaded model to your Python script for object detection.

#### Example Script

1. **Create a Python Virtual Environment**: Set up a virtual environment to manage dependencies.
2. **Install Dependencies**: Use the following command to install the necessary packages:

    ```bash
    pip install torch opencv-python git+https://github.com/ultralytics/yolov5
    ```

3. **Run Object Detection Script**: Execute the object detection script with the custom YOLO v5 models.

### Robotics Project Background

These scripts and models were developed as part of a robotics project aimed at automating the trimming of cannabis flowers. The envisioned solution included:

- **Two Robotic Arms**: One arm equipped with scissors for trimming, and the other for grasping the branch or bud.
- **Camera Integration**: A camera to detect the bud and guide the movements of both robotic arms.
- **Automated Trimming**: The camera detection and robotic arms coordination were designed to trim cannabis flowers automatically, without human involvement.

### Example Workflow

1. **Model Download and Setup**: Download the custom YOLO v5 models and set up your Python environment.
2. **Object Detection**: Use the provided script to perform object detection on images or live video.
3. **Integration with OAK-D Pro Camera**: Utilize the additional scripts to integrate the YOLO v5 models with the OAK-D Pro camera for enhanced functionality.

![Prototype](images/prototype_examples(1).PNG)


This repository provides a robust foundation for developing automated cannabis bud detection and trimming solutions, leveraging advanced image detection models and robotics integration.
