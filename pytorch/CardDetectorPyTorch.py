#!/usr/bin/python3

print("\nStarting PLaying Card Detector ...")
print("Loading system libs ...")
import cv2
import numpy
import torch
import torchvision
#import threading
import time
from utils import preprocess
import torch.nn.functional as F

import torchvision.transforms as transforms
from dataset import ImageClassificationDataset

import jetson.inference
import jetson.utils

import argparse
import sys

from pbn_file import PBN

print("Initializing diverter and motor controllers ...")
# Load diverter control
from dualmax import Motors
m_diverters = Motors()

from MotorControl import MotorConroller
myMotorController = MotorConroller()

############################################################################################
# Socket / keyboard I/O
############################################################################################
import selectors
import socket
import os
import fcntl

# set sys.stdin non-blocking
def set_input_nonblocking():
    orig_fl = fcntl.fcntl(sys.stdin, fcntl.F_GETFL)
    fcntl.fcntl(sys.stdin, fcntl.F_SETFL, orig_fl | os.O_NONBLOCK)

def create_socket(port, max_conn):
    host = socket.gethostname()  # get local machine name
    server_addr = (host, port)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.setblocking(False)
    server.bind(server_addr)
    server.listen(max_conn)
    return server

def read_from_client(conn, mask):
    global process_frames
    client_address = conn.getpeername()
    data = conn.recv(1024)
    print('Got {} request from {}'.format(data, client_address))
    parseCommandData(data)

def parseCommandData(data):
    if data == b'quit':
        quit()

    elif data.startswith(b'file', 0):
        # get hands from PBN file
        fileName = data.split(b' ',1)[1].decode()
        setPBN(fileName)
        init_session_data()

    elif data == b'deal':
        init_session_data()
        myMotorController.startMotors()

    elif not data:
         process_frames = False

def accept_client_connection(sock, mask):
    new_conn, addr = sock.accept()
    new_conn.setblocking(False)
    print('Connection from {}'.format(addr))
    m_selector.register(new_conn, selectors.EVENT_READ, read_from_client)

def quit():
    global process_frames
    print('Exiting...')
    process_frames = False

def setPBN(pbnFile):
    global player_S_cards
    if os.path.isfile(pbnFile):
        pbn_o = PBN(pbnFile)
        hands = pbn_o.get_hands()
        hnds_rs = pbn_o.get_hands_in_rs_format()
        print("player: S", hnds_rs['S'])
        player_S_cards = hnds_rs['S']

def read_from_keyboard(arg1, arg2):
    line = arg1.read()
    if line == 'quit\n':
        quit()
    else:
        print('User input: {}'.format(line))

# Initialize global per session data
def init_session_data():
    global m_card_deck
    global m_card_cnt
    global m_card_prediction
    global m_prediction

    m_card_deck = {}
    for category in dataset.categories:
        m_card_deck[category] = 0

    m_card_cnt=0
    m_card_prediction = 'NONE'
    m_prediction = ['NONE','NONE','NONE']
    


############################################################################################
#           Load optimizer and run evaluation
############################################################################################
def evaluate(nn_model, val_data):
    BATCH_SIZE = 8
    optimizer = torch.optim.Adam(nn_model.parameters())

    #try:
    train_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    time.sleep(1)

    i = 0
    sum_loss = 0.0
    error_count = 0.0
    for images, labels in iter(train_loader):
        # send data to device
        images = images.to(device)
        labels = labels.to(device)

        # execute model to get outputs
        outputs = nn_model(images)

        # compute loss
        loss = F.cross_entropy(outputs, labels)

        # update error count and progress
        error_count += len(torch.nonzero(outputs.argmax(1) - labels, as_tuple=False).flatten())
        count = len(labels.flatten())
        i += count
        sum_loss += float(loss)
        progress = i / len(val_data)
        loss = sum_loss / i
        accuracy = 1.0 - error_count / i

        print("{:.0%} Loss={:.6e} Accuracy={:.6e}".format(progress, loss, accuracy))
    
    print("Num Images={:d} Error Cnt={:e}".format(i, error_count))

print("Initializing socket server ...")
m_selector = selectors.DefaultSelector()

# listen to port 8080, at most 10 connections
m_server = create_socket(8080, 10)

m_selector.register(m_server, selectors.EVENT_READ, accept_client_connection)
m_selector.register(sys.stdin, selectors.EVENT_READ, read_from_keyboard)
print("done - listening to port 8080, at most 10 connections  ...")



# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.imageNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument('--pbn_file', type=str, default="demo1.pbn", help="PBN file to load")
parser.add_argument("--network", type=str, default="googlenet", help="pre-trained model to load (see below for options)")
parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
parser.add_argument("--width", type=int, default=224, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=224, help="desired height of camera stream (default is 720 pixels)")
parser.add_argument('--headless', action='store_true', default=(), help="run without display")
parser.add_argument('--run_val', default=False, action="store_true", help="run validation")
parser.add_argument('--gen_val_data', default=False, action="store_true", help="generate validation dataset")


is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)


print("Self Testing ..")
print(cv2.__version__)
print(numpy.__version__)

print(torch.__version__)
print(torchvision.__version__)

print('CUDA available: ' + str(torch.cuda.is_available()))
a = torch.cuda.FloatTensor(2).zero_()
print('Tensor a = ' + str(a))
b = torch.randn(2).cuda()
print('Tensor b = ' + str(b))
c = a + b
print('Tensor c = ' + str(c))


TASK = 'cards'
CATEGORIES = [ '2C','3C','4C','5C','6C','7C','8C','9C','10C','JC','QC','KC','AC',
               '2H','3H','4H','5H','6H','7H','8H','9H','10H','JH','QH','KH','AH',
               '2S','3S','4S','5S','6S','7S','8S','9S','10S','JS','QS','KS','AS',
               '2D','3D','4D','5D','6D','7D','8D','9D','10D','JD','QD','KD','AD','NONE']


DATASETS = ['A', 'B', 'C','V']
  
TRANSFORMS = transforms.Compose([
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

DATA_DIR = '/home/zbyszek/Desktop/pypro/data/card_detection_data/'

datasets = {}
for name in DATASETS:
    datasets[name] = ImageClassificationDataset(DATA_DIR + TASK + '_' + name, CATEGORIES, TRANSFORMS)

print("\nCard Categories:")    
print("{} task with {} categories defined".format(TASK, CATEGORIES))


# set default player S cards
player_S_cards = set(('2C','3C','4C','5C','6C','7C','8C','9C','10C','JH','QH','KH','AH'))

print("\nLoading player S hand from PBN file:", opt.pbn_file) 
# set player S hand from file
setPBN(opt.pbn_file)

# initialize active dataset
dataset = datasets[DATASETS[0]] 

# initialize val dataset
val_dataset = datasets[DATASETS[3]] 

# initialize collect dataset
collect_dataset = datasets[DATASETS[2]]

device = torch.device('cuda')
#device = torch.device('cpu')

############################################################################################
#           Load pretraind model and state form file 
############################################################################################

# ALEXNET
# model = torchvision.models.alexnet(pretrained=True)
# model.classifier[-1] = torch.nn.Linear(4096, len(dataset.categories))

# SQUEEZENET 
# model = torchvision.models.squeezenet1_1(pretrained=True)
# model.classifier[1] = torch.nn.Conv2d(512, len(dataset.categories), kernel_size=1)
# model.num_classes = len(dataset.categories)

# RESNET 18
m_model = torchvision.models.resnet18(pretrained=True)
m_model.fc = torch.nn.Linear(512, len(dataset.categories))

# RESNET 34
# model = torchvision.models.resnet34(pretrained=True)
# model.fc = torch.nn.Linear(512, len(dataset.categories))
    
m_model = m_model.to(device)
#m_model.load_state_dict(torch.load(DATA_DIR + 'my_card_model_v1.pth'))
m_model.load_state_dict(torch.load(DATA_DIR + 'epoch_2_BatchSize_64_LR_0_001.pth'))

# Set model to evaluation mode
m_model = m_model.eval()

# Validate against validation dataset
if opt.run_val:        
    evaluate(m_model, val_dataset)

# Initialize global data
init_session_data()
# m_card_deck = {}
# for category in dataset.categories:
#     m_card_deck[category] = 0

# m_card_cnt=0
# m_card_prediction = 'NONE'
# m_prediction = ['NONE','NONE','NONE']

# create video sources & outputs
m_input_video = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
m_output_video = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)
font = jetson.utils.cudaFont()


## Initialize calculated frame rate because it's calculated AFTER the first time it's displayed
frame_rate_calc = 1
freq = cv2.getTickFrequency()

cnt = 0

# process frames until the user exits
process_frames = True
while process_frames:
    
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # capture the next image
    cuda_img = m_input_video.Capture()

    # convert to numpy image
    img = jetson.utils.cudaToNumpy(cuda_img)

    # preprocess
    preprocessed = preprocess(img)
    #print(type(preprocessed))
    
    output = m_model(preprocessed)
    output = F.softmax(output, dim=1).detach().cpu().numpy().flatten()
    category_index = output.argmax()
    m_card_prediction = dataset.categories[category_index]

    #print(card_prediction)

    for i, score in enumerate(list(output)):
        if i==category_index:
            confidence = score
            if score > 0.97 :
                m_prediction[cnt%3] = m_card_prediction
                if m_prediction[0] == m_prediction[1] == m_prediction[2] and m_card_deck[m_card_prediction] == 0:
                    m_card_cnt = m_card_cnt + 1
                    if m_card_prediction != 'NONE':
                        #print("{:d} {:05.2f}% {:s}".format(m_card_cnt, confidence * 100, m_card_prediction)) 
                        print("{:05.2f}% {:s}".format(confidence * 100, m_card_prediction))
                    m_card_deck[m_card_prediction] += 1
                    if m_card_prediction in player_S_cards:
                        m_diverters.DivertTo()
                    else:
                        m_diverters.Bypass()
                    
                    # Save image for val 
                    if opt.gen_val_data:   
                        collect_dataset.save_entry(img, m_card_prediction)     

	# update the title bar
    m_output_video.SetStatus("Card Detector | {:.0f} FPS".format( int(frame_rate_calc)))

    # # overlay the result on the image
    font.OverlayText(cuda_img, cuda_img.width, cuda_img.height, "{:05.2f}% {:s}".format(confidence * 100, m_card_prediction), 5, 5, font.White, font.Gray40)
    #font.OverlayText(cuda_img, cuda_img.width, cuda_img.height, "{:s}".format("FPS:"+str(int(frame_rate_calc))), 5, 35, font.White, font.Gray40)

    # render the image
    m_output_video.Render(cuda_img)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    if cnt%60 == 0:
        frame_rate_calc = 1/time1
        
        # process socket and keyboard I/O
        for k, mask in m_selector.select(-1):
            callback = k.data
            callback(k.fileobj, mask)

        if myMotorController.isMotorRunning() and m_card_cnt >= 52:
            time.sleep(0.5)
            myMotorController.stopMotors()

    cnt = cnt + 1
	# exit on input/output EOS
    if not m_input_video.IsStreaming() or not m_output_video.IsStreaming():
        for category in dataset.categories:
            print(category, m_card_deck[category])
        break

# unregister events
m_selector.unregister(sys.stdin)

# close connection
m_server.shutdown(socket.SHUT_RDWR)
m_server.close()

#  close select
m_selector.close()

m_output_video.Close()
m_diverters.Cleanup()

myMotorController.stopMotors()


#import torch.onnx as onnx
#input_image = torch.zeros((1,3,224,224))
#onnx.export(model, input_image, 'model.onnx')