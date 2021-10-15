#!/usr/bin/python3
import cv2
import numpy
import torch
import torchvision
import threading
import time
from utils import preprocess
import torch.nn.functional as F

import torchvision.transforms as transforms
from dataset import ImageClassificationDataset

import jetson.inference
import jetson.utils

import argparse
import sys

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


print(cv2.__version__)
print(numpy.__version__)
print("Hello World")

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


DATASETS = ['A', 'B']
  
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
    
print("{} task with {} categories defined".format(TASK, CATEGORIES))


# initialize active dataset
dataset = datasets[DATASETS[0]] 

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
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, len(dataset.categories))

# RESNET 34
# model = torchvision.models.resnet34(pretrained=True)
# model.fc = torch.nn.Linear(512, len(dataset.categories))
    
model = model.to(device)
model.load_state_dict(torch.load(DATA_DIR + 'my_card_model.pth'))


############################################################################################
#           Load optimizer and run evaluation
############################################################################################

BATCH_SIZE = 8
optimizer = torch.optim.Adam(model.parameters())

#global BATCH_SIZE, LEARNING_RATE, MOMENTUM, model, dataset, optimizer, state

#try:
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

time.sleep(1)

model = model.eval()

i = 0
sum_loss = 0.0
error_count = 0.0
for images, labels in iter(train_loader):
    # send data to device
    images = images.to(device)
    labels = labels.to(device)

    # execute model to get outputs
    outputs = model(images)

    # compute loss
    loss = F.cross_entropy(outputs, labels)

    # increment progress
    #error_count += len(torch.nonzero(outputs.argmax(1) - labels).flatten())
    error_count += len(torch.nonzero(outputs.argmax(1) - labels, as_tuple=False).flatten())
    count = len(labels.flatten())
    i += count
    sum_loss += float(loss)
    progress = i / len(dataset)
    loss = sum_loss / i
    accuracy = 1.0 - error_count / i
    print("{:.0%} Loss={:.6e} Accuracy={:.0%}".format(progress,loss,accuracy))
        

card_deck = {}
for category in dataset.categories:
    card_deck[category] = 0


# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output1 = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)
font = jetson.utils.cudaFont()

## Initialize calculated frame rate because it's calculated AFTER the first time it's displayed
frame_rate_calc = 1
freq = cv2.getTickFrequency()

cnt = 0
# process frames until the user exits
while 1==1:
    
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # capture the next image
    cuda_img = input.Capture()

    # convert to numpy image
    img = jetson.utils.cudaToNumpy(cuda_img)

    # preprocess
    preprocessed = preprocess(img)

    #print(type(preprocessed))
    
    output = model(preprocessed)
    output = F.softmax(output, dim=1).detach().cpu().numpy().flatten()
    category_index = output.argmax()
    card_prediction = dataset.categories[category_index]
    
    #print(card_prediction)

    for i, score in enumerate(list(output)):
        if i==category_index:
            confidence = score
            if score > 0.95:
                card_deck[card_prediction] += 1
            

	# update the title bar
    output1.SetStatus("Card Detector | {:.0f} FPS".format( int(frame_rate_calc)))

    # # overlay the result on the image
    font.OverlayText(cuda_img, cuda_img.width, cuda_img.height, "{:05.2f}% {:s}".format(confidence * 100, card_prediction), 5, 5, font.White, font.Gray40)
    #font.OverlayText(cuda_img, cuda_img.width, cuda_img.height, "{:s}".format("FPS:"+str(int(frame_rate_calc))), 5, 35, font.White, font.Gray40)

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
        for category in dataset.categories:
            print(category, card_deck[category])
        break


#import torch.onnx as onnx
#input_image = torch.zeros((1,3,224,224))
#onnx.export(model, input_image, 'model.onnx')