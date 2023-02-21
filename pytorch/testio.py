#!/usr/bin/python3
import cv2
import numpy

import threading
import time


import jetson.inference
import jetson.utils

import argparse
import sys
from dualmax import Motors

motors = Motors()

# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.imageNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="googlenet", help="pre-trained model to load (see below for options)")
parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
parser.add_argument("--width", type=int, default=224, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=224, help="desired height of camera stream (default is 720 pixels)")
parser.add_argument('--headless', action='store_true', default=(), help="run without display")

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)


time.sleep(1)


# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output1 = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)
font = jetson.utils.cudaFont()

## Initialize calculated frame rate because it's calculated AFTER the first time it's displayed
frame_rate_calc = 1
freq = cv2.getTickFrequency()

cnt = 0
card_cnt=0
# process frames until the user exits
while 1==1:
    
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # capture the next image
    cuda_img = input.Capture()

    # convert to numpy image
    img = jetson.utils.cudaToNumpy(cuda_img)

    #if p[0] == p[1] == p[2] and card_deck[card_prediction] == 0 and motors.getFlowSensor1():
      

	# update the title bar
    output1.SetStatus("Card Detector | {:.0f} FPS".format( int(frame_rate_calc)))

    # render the image
    output1.Render(cuda_img)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    if cnt%60 == 0:
        frame_rate_calc = 1/time1

    cnt = cnt + 1
	# exit on input/output EOS
    if not input.IsStreaming() or not output1.IsStreaming():
        break

motors.Cleanup()