This repository contains some custom image detection models, trained with yolo v5. They detect cannabis buds (flower).
To use, download one of the models and add it to a python script for object detection. It will work with images or live video.
Once a detection occurs, a box is drawn around the flower.
There are also some scripts that will use the yolo v5 custom image detection models with the oak d pro camera.

To use: 
Create a python virtual environment then install dependencies and run object_detection script with custom yolov5 models.

pip install torch opencv-python git+https://github.com/ultralytics/yolov5

I was working on a robotics project that would have been able to trim cannabis flowers automatically, without human involvement. It would have been an automated trimming solution crafted specifically for the cannabis industry. I created these scripts and image detection models during the project.
It consisted of two robotic arms and a camera. One arm had a pair of scissors attached, they other would grasp a branch or bud and hold it while the scissor arm trimmed the extra leaves from the flower. The camera would detect the bud and direct the movement of both arms using code.

However, the rich kids that run the cannabis industry in arizona decided to exile me from the industry before I could complete my work. They didn't want me to make more money than their parents gave them. 