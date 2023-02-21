#import pigpio
import RPi.GPIO as GPIO
import time

# _pi = pigpio.pi()
# if not _pi.connected:
#     raise IOError("Can't connect to pigpio")

# Motor speeds are specified as numbers between -MAX_SPEED and
# MAX_SPEED, inclusive.
_max_speed = 100
MAX_SPEED = _max_speed

# Frequency Hz
_freq_PWM = 1000

# Jetson
# Pre-test configuration, if boot-time pinmux doesn't set up PWM pins:
# Set BOARD pin 32 as mux function PWM (func 1):
# sudo busybox devmem 0x2430040 32 0x401
# Set BOARD pin 33 as mux function PWM (func 2):
# sudo busybox devmem 0x2440020 32 0x402

# Jetson board pins assignment
_pin_nEN = 29
_pin_nFAULT = 31
_pin_M1PWM = 32
_pin_M2PWM = 33
_pin_M1DIR = 18
_pin_M2DIR = 22
_pin_FLOW_SENSOR1 = 36
_pin_FLOW_SENSOR2 = 38

class Motor(object):
    MAX_SPEED = _max_speed

    def __init__(self, pwm_pin, dir_pin):
        self.pwm_pin = pwm_pin
        self.dir_pin = dir_pin
        GPIO.setmode(GPIO.BOARD)  # Board pin-numbering scheme from Raspberry Pi
        
        # set dir_pin pin as an output pin with optional initial state of HIGH
        GPIO.setup(self.dir_pin, GPIO.OUT, initial=GPIO.LOW)

        # set pin as an output pin with optional initial state of LOW
        GPIO.setup(self.pwm_pin, GPIO.OUT, initial=GPIO.LOW)
        self.p = GPIO.PWM(self.pwm_pin, _freq_PWM)
        dutyCycle = 0
        self.p.start(dutyCycle)

    def setSpeed(self, speed):
        if speed < 0:
            speed = -speed
            dir_value = 1
        else:
            dir_value = 0

        if speed > MAX_SPEED:
            speed = MAX_SPEED

        GPIO.output(self.dir_pin, dir_value)
        self.p.ChangeDutyCycle(int(speed)) #* 6250 / 3))

class Motors(object):
    MAX_SPEED = _max_speed
    count = 0
    diverter1_state = 0
    diverter1_flip = True

    def __init__(self):

        self.diverter1_state = 0
        self.diverter1_flip = True

        # Set board pin-numbering scheme from Raspberry Pi
        GPIO.setmode(GPIO.BOARD)
        
        self.motor1 = Motor(_pin_M1PWM, _pin_M1DIR)
        self.motor2 = Motor(_pin_M2PWM, _pin_M2DIR)
        
        # set _pin_nEN pin as an output pin with optional initial state of HIGH
        # to disable drivers by default
        GPIO.setup(_pin_nEN, GPIO.OUT, initial=GPIO.HIGH)

        self.setSpeeds(65, 65)
        
        # set _pin_nFAULT pin as an input pin
        GPIO.setup(_pin_nFAULT, GPIO.IN)  

        # set _pin_FLOW_SENSOR1 pin as an input pin
        GPIO.setup(_pin_FLOW_SENSOR1, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)  

        # set _pin_FLOW_SENSOR2 up as input 
        GPIO.setup(_pin_FLOW_SENSOR2, GPIO.IN, pull_up_down=GPIO.PUD_UP) 

        count = 0
        # The GPIO.add_event_detect()   
        # when a rising edge is detected on sensor2  
        GPIO.add_event_detect(_pin_FLOW_SENSOR1, GPIO.RISING, callback=self, bouncetime=1) 

    def __call__(self, *args):
        self.my_callback(_pin_FLOW_SENSOR1)
        return

    def setSpeeds(self, m1_speed, m2_speed):
        self.motor1.setSpeed(m1_speed)
        self.motor2.setSpeed(m2_speed)

    def getFault(self):
        return not GPIO.input(_pin_nFAULT)

    def getFlowSensor1(self):
        return not GPIO.input(_pin_FLOW_SENSOR1)

    def enable(self):
        GPIO.output(_pin_nEN, GPIO.LOW)

    def disable(self):
        GPIO.output(_pin_nEN, GPIO.HIGH)

    def forceStop(self):
        self.disable()
        self.setSpeeds(0, 0)

    
    def Cleanup(self):
        GPIO.cleanup()

    def Bypass(self):
        if self.diverter1_state == 1:
            self.diverter1_state = 0
            self.diverter1_flip = True
    
    def DivertTo(self):
        if self.diverter1_state == 0:
            self.diverter1_state = 1
            self.diverter1_flip = True



    # now we'll define the threaded callback function  
    # this will run in another thread when our event is detected  
    def my_callback(self, channel):  
        if GPIO.input(channel):
            if self.diverter1_flip:
                self.diverter1_flip = False
                #print ("Rising edge detected on sensor1 count={:d}", self.count)
                if self.diverter1_state == 1:
                    #self.setSpeeds(50, 50)
                    GPIO.output(_pin_M1DIR, 0)
                else:
                    GPIO.output(_pin_M1DIR, 1)
                
                #self.enable() 
                GPIO.output(_pin_nEN, GPIO.LOW)              
                time.sleep(0.03)
                #self.forceStop()
                GPIO.output(_pin_nEN, GPIO.HIGH)
                

# motors = Motors()
# motors.setSpeeds(50, 50)
# motors.enable()
# time.sleep(0.10)
# motors.setSpeeds(-50, -50)
# time.sleep(0.10)
# print(motors.getFlowSensor1())

# motors.forceStop()
# motors.Cleanup()


