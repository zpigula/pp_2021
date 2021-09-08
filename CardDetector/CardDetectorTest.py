############## Python-OpenCV Playing Card Detector
#
# Description: Python script to detect and identify playing cards
# from a PiCamera video feed.
#

# Import necessary packages
import cv2
import numpy as np
import time
import os
import CardsTest
import MotorControl
#import VideoStream

# import CapWebcam

# import matplotlib.pyplot as plt

### ---- INITIALIZATION ---- ###
# Define constants and initialize variables

## Camera settings
# IM_WIDTH = 1280
# IM_HEIGHT = 720
IM_WIDTH = 256
IM_HEIGHT = 800

FRAME_RATE = 15

width=1280
height=960
flip=2

#camSet='nvarguscamerasrc sensor-id=0  ! video/x-raw(memory:NVMM), with=3264, height=2464, framerate=21/1, format=NV12  ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(width)+', height='+str(height)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
camSet='nvarguscamerasrc sensor-id=0 tnr-mode=2 tnr-strength=1 wbmode=3 ! video/x-raw(memory:NVMM), with=3264, height=2464, framerate=21/1, format=NV12  ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(width)+', height='+str(height)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! videobalance ! appsink'
videostream=cv2.VideoCapture(camSet)

## Initialize calculated frame rate because it's calculated AFTER the first time it's displayed
frame_rate_calc = 1
freq = cv2.getTickFrequency()

## Define font to use
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize camera object and video feed from the camera. The video stream is set up
# as a seperate thread that constantly grabs frames from the camera feed. 
# See VideoStream.py for VideoStream class definition
## IF USING USB CAMERA INSTEAD OF PICAMERA,
## CHANGE THE THIRD ARGUMENT FROM 1 TO 2 IN THE FOLLOWING LINE:
# videostream = VideoStream.VideoStream((IM_WIDTH,IM_HEIGHT),FRAME_RATE,1,0).start()

# videostream = VideoStream.VideoStream((IM_WIDTH,IM_HEIGHT),FRAME_RATE,2,0).start()

path = os.path.dirname(os.path.abspath(__file__))
# videostream = cv2.VideoCapture(path + '/Test_Videos/video.h264')
# videostream = cv2.VideoCapture(path + '/Test_Videos/videoTest.avi')

time.sleep(1)  # Give the camera time to warm up

# Load the train rank and suit images
# path = os.path.dirname(os.path.abspath(__file__))
train_ranks = CardsTest.load_ranks(path + '/Card_Imgs/')
train_suits = CardsTest.load_suits(path + '/Card_Imgs/')

### ---- MAIN LOOP ---- ###
# The main loop repeatedly grabs frames from the video stream
# and processes them to find and identify playing cards.

cam_quit = 0  # Loop control variable
ret = True
card_rank = "Unknown"
card_suit = "Unknown"
count = 0

print("Bridge Card Dealing Machine Ver 0.5")

# initialize deck
deck = CardsTest.deck_init()
#for playing_card in deck:
#    print (playing_card, deck[playing_card])

print("Initializing Motors ...")
MotorControl.energize('00275197')
MotorControl.energize('00275128')

MotorControl.targetVelocity('00275197', 150000000)
MotorControl.targetVelocity('00275128', 150000000)

# MotorControl.stop('00275197')
# MotorControl.stop('00275128')
# MotorControl.deEnergize('00275197')
# MotorControl.deEnergize('00275128')


print("Starting Motors ...")
# Begin capturing frames
while (cam_quit == 0 and ret):

    # Grab frame from video stream
    ret, in_image = videostream.read()
    #image = videostream.read()
    if ret:

        #cv2.imshow("image", image)

        #cv2.moveWindow("image", 0, 400)
        # cv2.waitKey(1)

        # plt.imshow(image, cmap='gray', interpolation='bicubic')
        # plt.show()


        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        image = in_image[400:400 + 512, 460:460 + 360]

        # Pre-process camera image (gray, blur, and threshold it)
        pre_proc1 = CardsTest.preprocess_image(image)
        pre_proc = pre_proc1.copy()
        # Find contours and sort them by size
        cnts, hier = cv2.findContours(pre_proc1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        if len(cnts) == 0:
            print('No contours found!')
            quit()

        # Initialize a new "cards" list to assign the card objects.
        # k indexes the newly made array of cards.
        cards = []
        k = 0

        #cv2.imshow("pre_proc", pre_proc)
        #cv2.moveWindow("pre_proc", 500, 200)
        # cv2.waitKey(1)

        if len(cnts) > 2:
            rank = cnts[1]

            # Approximate the corner points of the card
            peri = cv2.arcLength(rank, True)
            approx = cv2.approxPolyDP(rank, 0.01 * peri, True)
            pts = np.float32(approx)

            x, y, w, h = cv2.boundingRect(rank)

            suit = cnts[2]

            # Approximate the corner points of the card
            peri = cv2.arcLength(suit, True)
            approx = cv2.approxPolyDP(suit, 0.01 * peri, True)
            pts = np.float32(approx)

            x1, y1, w1, h1 = cv2.boundingRect(suit)

            if h < h1 :
                x2 , y2, w2, h2 = x, y, w, h
                x, y, w, h = x1, y1, w1, h1
                x1, y1, w1, h1 = x2, y2, w2, h2

            rank_image = pre_proc[y:y + h, x:x + w]
            # rank_image = cv2.cvtColor(rank_image, cv2.COLOR_BGR2GRAY)
            #cv2.imshow("rank_image", rank_image)
            #cv2.moveWindow("rank_image", 1800, 200)
            # cv2.waitKey(1)

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            rank_zoom = cv2.resize(rank_image, (0, 0), fx=4, fy=4)
            rank_roi = rank_zoom  # Grabs portion of image that shows rank
            rank_sized = cv2.resize(rank_roi, (CardsTest.RANK_WIDTH, CardsTest.RANK_HEIGHT), 0, 0)
            final_img = rank_sized
            suit_image = pre_proc[y1:y1 + h1, x1:x1 + w1]
            cv2.rectangle(image, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)

            # cv2.imshow("image", image)
            # cv2.waitKey(1)

            suit_zoom = cv2.resize(suit_image, (0, 0), fx=4, fy=4)
            suit_roi = suit_zoom  # Grabs portion of image that shows rank
            suit_sized = cv2.resize(suit_roi, (CardsTest.SUIT_WIDTH, CardsTest.SUIT_HEIGHT), 0, 0)

            cards.append(CardsTest.create_card(rank_sized, suit_sized))

            # Find the best rank and suit match for the card.
            cards[k].best_rank_match, cards[k].best_suit_match, cards[k].rank_diff, cards[
                k].suit_diff = CardsTest.match_card(cards[k], train_ranks, train_suits)
            
            # print(count, cards[k].best_rank_match, cards[k].best_suit_match, cards[k].rank_diff, cards[k].suit_diff)
            
            if cards[k].best_rank_match != "Unknown" and cards[k].best_suit_match != "Unknown":
                image = CardsTest.draw_results(image, cards[k])
                #MotorControl.getInputs('00275128')
                if cards[k].best_rank_match != card_rank or cards[k].best_suit_match != card_suit:
                    count = count + 1
                    print(count, cards[k].best_rank_match, cards[k].best_suit_match, cards[k].rank_diff, cards[k].suit_diff)
                    deck[cards[k].best_rank_match+cards[k].best_suit_match] += 1
                    # if count == 31 or count == 33 or count == 42:
                    #     print(count)
                    #if count<52:
                         #MotorControl.motorRun('00275197', 4428000)
                         #MotorControl.targetVelocity('00275197', 10000000)
                         #MotorControl.motorRun('00275128', 28000)
                         #MotorControl.targetVelocity('00275128', 10000000)
                         
                card_rank = cards[k].best_rank_match
                card_suit = cards[k].best_suit_match

        # Draw frame rate in the corner of the image. Frame rate is calculated at the end of the main loop,
        # so the first time this runs, frame rate will be shown as 0.
        cv2.putText(image, "FPS: " + str(int(frame_rate_calc)), (10, 26), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

        # Finally, display the image with the identified cards!
        cv2.imshow("Card Detector", image)
        # cv2.resizeWindow("Card Detector", 600, 400)
        cv2.moveWindow("Card Detector", 100, 50)
        # Calculate frame rate
        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1

        # Poll the keyboard. If 'q' is pressed, exit the main loop.

        key = cv2.waitKey(1)
        if key == ord('q'):
            cam_quit = 1
        if key == ord('n'):
            MotorControl.motorRun('00275197', 4428000)
            MotorControl.targetVelocity('00275197', 200000000)
            MotorControl.motorRun('00275128', 28000)
            MotorControl.targetVelocity('00275128', 200000000)

# Close all windows and close the PiCamera video stream.
for playing_card in deck:
    if deck[playing_card] == 0:
         print ("Missing", playing_card)
    
MotorControl.stop('00275197')
MotorControl.stop('00275128')
MotorControl.deEnergize('00275197')
MotorControl.deEnergize('00275128')

videostream.release()
cv2.destroyAllWindows

#cv2.destroyAllWindows()
# videostream.release()
# videostream.stop()
