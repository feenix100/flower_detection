#!/usr/bin/env python3

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import argparse
from servo import ServoController

#servo script contains functions for controlling servos
#opens and closes scissors when detection occurs
#uses yolov8 model to detect bud
#chat gpt 3.5(freeVersion) may have been involved in the development of this code


servo_controller = ServoController(servo_channel=0, min_angle=74, max_angle=178, servo_speed=2, direction=1)


''' Spatial Detection with YoloV8 example
This script will control a pair of scissors attached to a robot arm when a bud is detected within a specified range(400mm) of the camera. The scissors will open and close when the bud is detected in range.


Use tools.luxonis.com to convert your yolov5 custom model to a blob file that can be used by Oak D. Set input image shape to 416 and shaves to 5 for best results. The converter will put a few files in a zip for you to download. 
I used Oak d pro connected to raspberry pi 8gb with custom yolov5 model converted to blob to run this script.

  Performs inference on RGB camera and retrieves spatial location coordinates: x,y,z relative to the center of depth map.
  Can be used for custom yolov5 networks change nnBlobPath to your custom blob path.

I used the spatial_tiny_yolo script and modified a few lines to work with yoloV8. I made sure to change the spatialDetectionNetwork.setAnchorMasks and spatialDetectionNetwork.setAnchors to match the info in my json file obtained from blob zip downloaded from tools.luxonis.com. Lines 92 and 93
 '''

parser = argparse.ArgumentParser(description="Spatial yoloV8 example")
parser.add_argument("--model", type=str, default="nnBlobPath", help="Choose the model or provide custom path")

args = parser.parse_args()

# Get argument first
nnBlobPath = str((Path(__file__).parent / Path('models/yolov8/bud69epoch_openvino_2022.1_10shave.blob')).resolve().absolute())
if 1 < len(sys.argv):
    arg = sys.argv[1]
else:
    print("Using YoloV8 model. To use other yolos/mobilenet, use a different script")

if not Path(nnBlobPath).exists():
    import sys
    raise FileNotFoundError('The path to your blob file is incorrect, nnBlobPath')

syncNN = True

def run_scissors(distance_mm):
    
    # scissors are approx 5in(120mm)12cm from camera
    # this function will open and close the scissors when the detected leaf is within 400 mm of camera
    # the scissors and camera should be in a fixed and known position
    if distance_mm <= 900: # distance between camera and scissors
        servo_controller.move_servo()

       

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
#spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
#stereo = pipeline.create(dai.node.StereoDepth)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)

nnNetworkOut = pipeline.create(dai.node.XLinkOut)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutNN = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")
xoutDepth.setStreamName("depth")
nnNetworkOut.setStreamName("nnNetwork")

# Properties
camRgb.setPreviewSize(416, 416)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

#stereoDepth node calculates disparity and or depth from the stereo camera pair
stereo = pipeline.create(dai.node.StereoDepth)

# setting node configs changes threshold 200 for Accuracy, 245 for density
#stereo.initialConfig.setConfidenceThreshold(threshold)
#stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)


# Align depth map to the perspective of RGB camera, on which inference is done
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())
stereo.setSubpixel(True)


spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)

spatialDetectionNetwork.setBlobPath(nnBlobPath)

spatialDetectionNetwork.setConfidenceThreshold(0.5)

spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(5000)

# yolo v5 label texts
labelMap = ["bud"]


# Yolo specific parameters
spatialDetectionNetwork.setNumClasses(1)
spatialDetectionNetwork.setCoordinateSize(4)
spatialDetectionNetwork.setAnchors([10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0])
spatialDetectionNetwork.setAnchorMasks({ "side52": [0,1,2], "side26": [3,4,5], "side13": [6,7,8] })
spatialDetectionNetwork.setIouThreshold(0.5)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

camRgb.preview.link(spatialDetectionNetwork.input)
if syncNN:
    spatialDetectionNetwork.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

spatialDetectionNetwork.out.link(xoutNN.input)

stereo.depth.link(spatialDetectionNetwork.inputDepth)
spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)
spatialDetectionNetwork.outNetwork.link(nnNetworkOut.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    networkQueue = device.getOutputQueue(name="nnNetwork", maxSize=4, blocking=False);

    startTime = time.monotonic()
    counter = 0
    fps = 0
    color = (255, 255, 255)
    printOutputLayersOnce = True

    while True:
        inPreview = previewQueue.get()
        inDet = detectionNNQueue.get()
        depth = depthQueue.get()
        inNN = networkQueue.get()

        #if printOutputLayersOnce:
        #   toPrint = 'Output layer names:'
        #    for ten in inNN.getAllLayerNames():
        #        toPrint = f'{toPrint} {ten},'
        #    print(toPrint)
        #    printOutputLayersOnce = False;

        frame = inPreview.getCvFrame()
        depthFrame = depth.getFrame() # depthFrame values are in millimeters

        depth_downscaled = depthFrame[::4]
        min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
        max_depth = np.percentile(depth_downscaled, 99)
        depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        detections = inDet.detections

        # If the frame is available, draw bounding boxes on it and show the frame
        height = frame.shape[0]
        width  = frame.shape[1]

        detection_occured = False #initialize detection flag

        if len(detections) > 0:
            detection_occurred = True
        else:
            detection_occurred = False

        for detection in detections:
            roiData = detection.boundingBoxMapping
            roi = roiData.roi
            roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
            topLeft = roi.topLeft()
            bottomRight = roi.bottomRight()
            xmin = int(topLeft.x)
            ymin = int(topLeft.y)
            xmax = int(bottomRight.x)
            ymax = int(bottomRight.y)
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, 1)
            
            #get the z distance between camera and detected object
            distance_mm = detection.spatialCoordinates.z
            
            # Denormalize bounding box
            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)
        
            
           #call the function that will open and close the scissors held by the robot claw
            if detection_occurred:
                run_scissors(distance_mm)
           
            try:
                label = labelMap[detection.label]
            except:
                label = detection.label
                
            cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
       # cv2.imshow("depth", depthFrameColor)
        cv2.imshow("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break
