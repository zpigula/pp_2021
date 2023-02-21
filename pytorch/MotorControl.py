# Motor Control Lib

# 00275197,         Tic T249 Stepper Motor Controller            
# 00275128,         Tic T249 Stepper Motor Controller    

# Uses ticcmd to send and receive data from the Tic over USB.
# Works with either Python 2 or Python 3.
#
# NOTE: The Tic's control mode must be "Serial / I2C / USB".
 
import subprocess
import yaml
 
def ticcmd(*args):
  return subprocess.check_output(['ticcmd'] + list(args))

def motorRun(motorId, relPos): 
  status = yaml.load(ticcmd('-s', '-d', str(motorId), '--full'), Loader=yaml.FullLoader)
 
  position = status['Current position']
  # print("Current position is {}.".format(position))
 
  new_target = position + relPos
  # print("Setting target position to {}.".format(new_target))
  ticcmd('-d', str(motorId), '--exit-safe-start', '--position', str(new_target))


def getInputs(motorId): 
  status = ticcmd('-s', '-d', str(motorId))
 
  #position = status['Current position']
  print("Current position is {}.",status)

def targetVelocity(motorId, vel): 
  ticcmd('-d', str(motorId), '--exit-safe-start', '--velocity', str(vel))

def stop(motorId): 
  ticcmd('-d', str(motorId), '--enter-safe-start')

def energize(motorId): 
  ticcmd('-d', str(motorId), '--energize')

def deEnergize(motorId): 
  ticcmd('-d', str(motorId), '--deenergize')



class MotorConroller(object):

  transportMotor = '00275197'
  feederMotor = '00275128'
  feederMotorVelocity = 350000000
  transportMotorVelocity = 350000000
  isRunning = False

  def startMotors(self): 
    self.isRunning = True
    energize(self.transportMotor)
    energize(self.feederMotor)

    targetVelocity(self.transportMotor, self.transportMotorVelocity)
    targetVelocity(self.feederMotor, self.feederMotorVelocity)

  def stopMotors(self): 
    stop(self.transportMotor)
    stop(self.feederMotor)
    deEnergize(self.transportMotor)
    deEnergize(self.feederMotor)
    self.isRunning = False 
  
  def isMotorRunning(self):
    return self.isRunning

