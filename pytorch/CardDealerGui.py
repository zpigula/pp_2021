import subprocess
import sys
import time
from tkinter import *
from collections import namedtuple
from MotorControl import MotorConroller
from pbn_file import PBN

import socket
mySocket = socket.socket()

myMotorController = MotorConroller()

# Create client socket
def client_connect():
    try:
        host = socket.gethostname()  # get local machine name
        port = 8080  # Make sure it's within the > 1024 $$ <65535 range  
        mySocket.connect((host, port))
        myConnectButton["state"]=DISABLED
        myShutdownButton["state"]=NORMAL
    except:
        print("no connection to card detector app ...")

def quit():
    global mySocket
    message = b'quit'  
    mySocket.send(message)
    myRunButton["state"] = NORMAL
    myConnectButton["state"]=DISABLED
    myShutdownButton["state"]=DISABLED
    mySocket.close()
    mySocket = socket.socket()

def switchFile(file):
    if myShutdownButton["state"]==NORMAL:
        global mySocket
        file = "file " + file
        message = file.encode('utf-8')
        mySocket.send(message)

def deal():
    if myShutdownButton["state"]==NORMAL:
        global mySocket       
        message = b'deal' 
        mySocket.send(message)

def run_pt_script():
    global run_val
    global gen_dataset

    command = 'python3 /home/zbyszek/pp_repo/pytorch/CardDetectorPyTorch.py csi://0 --input-width=224 --input-height=224'
    #command = 'python3 /home/zbyszek/pp_repo/pytorch/CardDetectorPyTorch.py csi://0 file://my_video.mp4 --input-width=224 --input-height=224'

    
    if run_val.get():
        command += " --run_val"

    if gen_dataset.get():
        command += " --gen_val_data"
    
    pbn_filename = " --pbn_file " + myListBox.get(myListBox.curselection())
    command = command + pbn_filename
    print("starting: ", command)

    subprocess.Popen(command, shell = True)  # Run other script - doesn't wait for it to finish.
    myRunButton["state"] = DISABLED
    myConnectButton["state"]=NORMAL
    myShutdownButton["state"]=DISABLED


def myClick():

    root.update()

    # define text file to open
    fileName = myListBox.get(myListBox.curselection())

    # get hands from PBN file
    pbn_o = PBN(fileName)
    hands = pbn_o.get_hands()
    hnds_rs = pbn_o.get_hands_in_rs_format()

    switchFile(fileName)

    #myLabel3 = Label(root, text="hands: " + hands['N']['S'])
    #myLabel3.pack()
    myPlayerN = Label(root, padx=15, compound = LEFT, justify=RIGHT, text= " " + hands['N']['S'] + "\n " + hands['N']['H'] + "\n " + hands['N']['D'] + "\n " + hands['N']['C'], image=suitsImg , font = "Helvetica 16 italic")
    myPlayerS = Label(root, padx=15, compound = LEFT, justify=RIGHT, text= " " + hands['S']['S'] + "\n " + hands['S']['H'] + "\n " + hands['S']['D'] + "\n " + hands['S']['C'], image=suitsImg , font = "Helvetica 16 bold italic" )
    myPlayerE = Label(root, padx=15, compound = LEFT, justify=RIGHT, text= " " + hands['E']['S'] + "\n " + hands['E']['H'] + "\n " + hands['E']['D'] + "\n " + hands['E']['C'], image=suitsImg , font = "Helvetica 16 italic" )
    myPlayerW = Label(root, padx=15, compound = LEFT, justify=RIGHT, text= " " + hands['W']['S'] + "\n " + hands['W']['H'] + "\n " + hands['W']['D'] + "\n " + hands['W']['C'], image=suitsImg , font = "Helvetica 16 italic" )

    myPlayerN_label = Label(root, padx=10, pady=10, justify=CENTER, text= "N", font = "Helvetica 16 bold")
    myPlayerS_label = Label(root, padx=10, pady=10, justify=CENTER, text= "S", font = "Helvetica 16 bold" )
    myPlayerE_label = Label(root, padx=20, justify=RIGHT, text= "E", font = "Helvetica 16 bold" )
    myPlayerW_label = Label(root, padx=20, justify=LEFT, text= "W", font = "Helvetica 16 bold" )

    #myPlayerN.pack()
    #myPlayerS.pack()
    myPlayerN.grid(row=1,column=2)
    myPlayerS.grid(row=3,column=2)
    myPlayerE.grid(row=2,column=3)
    myPlayerW.grid(row=2,column=1)

    myPlayerN_label.grid(row=2,column=2, sticky=N)
    myPlayerS_label.grid(row=2,column=2, sticky=S)
    myPlayerE_label.grid(row=2,column=2, sticky=E)
    myPlayerW_label.grid(row=2,column=2, sticky=W)
    # myLabel3.grid(row=0,column=5)

def showSelected(event):
    myClick()

root = Tk()
content = Frame(root)
#lf = LabelFrame(root, text='Alignment')
#lf.grid(column=0, row=0, padx=20, pady=20)

frame = LabelFrame(content, text='Machine Control', padx = 5, pady = 5, width=400)
frame1 = LabelFrame(content, text='DNN Model / Card Classification Control', padx = 5, pady = 5)
# grid layout for the input frame
frame1.columnconfigure(0, weight=1)
frame1.columnconfigure(1, weight=3)

root.title("My Bridge Machine")
#root.iconbitmap('C:/Users/Zbyszek/Documents/DarSerca/images/ikony/pdf_icon.jpg')
suitsImg = PhotoImage(file="suits.png")

#e=Entry(root)
#e.grid(row=0,column=1, columnspan=2)
#textGet = e.get()
c_padx=10
c_pady=5
b_padx=20

myListBox = Listbox(frame, height=3)
myListBox.insert(1,"demo1.pbn")
myListBox.insert(2,"demo2.pbn")
myListBox.insert(3,"demo3.pbn")
myListBox.grid(row=0, column=0, columnspan=1, rowspan=2, padx=c_padx, pady=c_pady)
myListBox.bind('<<ListboxSelect>>', showSelected)

myStartButton = Button(frame,text="Deal", padx=b_padx, command=deal)
myStartButton.grid(row=0, column=1, padx=c_padx, pady=c_pady)
myStopButton = Button(frame,text="Stop", padx=b_padx, command=myMotorController.stopMotors)
myStopButton.grid(row=1, column=1, padx=c_padx, pady=c_pady)

myRunButton = Button(frame1,text="Open", padx=b_padx, width=4, command=run_pt_script)
myRunButton.grid(row=0, column=1, padx=c_padx, pady=c_pady,sticky=E)
myConnectButton = Button(frame1,text="Connect", padx=b_padx, width=4, command=client_connect)
myConnectButton.grid(row=1, column=1, padx=c_padx, pady=c_pady,sticky=E)
myShutdownButton = Button(frame1,text="Shutdown", padx=b_padx, width=4, command=quit)
myShutdownButton.grid(row=2, column=1, padx=c_padx, pady=c_pady,sticky=E)

run_val = BooleanVar()
myRunValCheck = Checkbutton(frame1,
                text='Validate',
                variable=run_val,
                onvalue=True,
                offvalue=False)

myRunValCheck.grid(row=0, column=0, columnspan=1, sticky=(W))

gen_dataset = BooleanVar()
myGenDataCheck = Checkbutton(frame1,
                text='Collect Data',
                variable=gen_dataset,
                onvalue=True,
                offvalue=False)

myGenDataCheck.grid(row=1, column=0, columnspan=1, sticky=(W))

myConnectButton["state"]=DISABLED
myShutdownButton["state"]=DISABLED

content.grid(column=0, row=0,rowspan=4, padx=10, pady=10, sticky=(N))
frame.grid(column=0, row=0, padx=10, pady=10)
frame1.grid(column=0, row=1, padx=10, pady=10, sticky=(E,W))
# start card detector
#run_pt_script()
#time.sleep(10)

# populate card data
myListBox.select_set(0)
myClick()

#client_connect()

# run gui loop
root.mainloop()

# shutdown
myMotorController.stopMotors()
#quit()
time.sleep(2)
mySocket.close()
print("EXIT")


