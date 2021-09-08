############## Playing Card Machine ###############
# Start machine transport motors
#

# Import necessary packages
import cv2
import numpy as np
import time
import os
import Cards
import CardsTest
import MotorControl
import VideoStream



print("Initializing Motors ...")
MotorControl.energize('00275197')
MotorControl.energize('00275128')

MotorControl.targetVelocity('00275197', 1500000)
MotorControl.targetVelocity('00275128', 1500000)

# MotorControl.stop('00275197')
# MotorControl.stop('00275128')
# MotorControl.deEnergize('00275197')
# MotorControl.deEnergize('00275128')


print("Starting Motors ...")


MotorControl.motorRun('00275197', 4428000)
MotorControl.motorRun('00275128', 2800000)


# cam_quit = 0
# while (cam_quit == 0):
#     # Poll the keyboard. If 'q' is pressed, exit the main loop.
#     key = cv2.waitKey(100) & 0xFF
#     if key == ord("q"):
#         cam_quit = 1
