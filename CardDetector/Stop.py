# Stop motors
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


    
MotorControl.stop('00275197')
MotorControl.stop('00275128')
MotorControl.deEnergize('00275197')
MotorControl.deEnergize('00275128')

