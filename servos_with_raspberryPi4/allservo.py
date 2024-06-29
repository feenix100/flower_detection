import tkinter as tk
from tkinter import Scale, Button
from adafruit_servokit import ServoKit

#this script was created with chatgpt 3.5 september 2023 free version.
#this script will control servos attached to a pwm 16 servo hat from adafruit
#it creates a gui using tkinter with a button and slider for each servo
#also has an exit button that will close the script
# i am currently using a raspberry pi 4 with 64 bit os and 4 different pwm servos

# Initialize the ServoKit
kit = ServoKit(channels=16)  # Use the default I2C address

# Specify the servo channels
servo_channels = [0, 4, 8, 12]

# Create a function to exit the application
def exit_app():
    root.destroy()

# Create a GUI
root = tk.Tk()
root.title("Servo Control")

servo_sliders = []

def update_servo(servo_num):
    angle = servo_sliders[servo_num].get()
    kit.servo[servo_channels[servo_num]].angle = angle

# Create a function to handle slider updates
def slider_update(servo_num):
    return lambda event: update_servo(servo_num)
#create the buttons and sliders for servos
for i in range(4):
    servo_frame = tk.Frame(root)
    servo_frame.pack(side=tk.LEFT, padx=20, pady=20)
    servo_label = tk.Label(servo_frame, text=f"Servo {i+1}")
    servo_label.pack()
    
    slider = Scale(servo_frame, from_=0, to=180, orient=tk.HORIZONTAL, command=slider_update(i))
    
    slider.set(90)  # Set the initial position to 90 degrees
    slider.pack()
    servo_sliders.append(slider)

# Add an "Exit" button
exit_button = Button(root, text="Exit", command=exit_app)
exit_button.pack()

# Run the GUI
root.mainloop()
