############## Motor Controller Module ###############

# Import necessary packages
import cv2
import MotorControl

transportMotor = '00275197'
feederMotor = '00275128'
feederMotorVelocity = 350000000
transportMotorVelocity = 250000000

def startMotors(): 
    MotorControl.energize(transportMotor)
    MotorControl.energize(feederMotor)

    MotorControl.targetVelocity(transportMotor, transportMotorVelocity)
    MotorControl.targetVelocity(feederMotor, feederMotorVelocity)

def stopMotors(): 
    MotorControl.stop(transportMotor)
    MotorControl.stop(feederMotor)
    MotorControl.deEnergize(transportMotor)
    MotorControl.deEnergize(feederMotor)


quit = 0
print("Starting Motor Controller ...")
while (quit == 0):

    #key = cv2.waitKey(0)

    key = input("Press s/q to stop motors or r to run...")
    if key == "q":
        quit = 1
    if key == "s":
        stopMotors()
    if key == "r":
        startMotors()
    

stopMotors


cv2.destroyAllWindows

