from dualmax import Motors

motors = Motors()

# _pi = pigpio.pi()
# if not _pi.connected:
#     raise IOError("Can't connect to pigpio")

# Motor speeds for this library are specified as numbers between -MAX_SPEED and
# MAX_SPEED, inclusive.
# This has a value of 480 for historical reasons/to maintain compatibility with
# older libraries for other Pololu boards (which used WiringPi to set up the
# hardware PWM directly).
_max_speed = 480
MAX_SPEED = _max_speed

# Jetson
# Pre-test configuration, if boot-time pinmux doesn't set up PWM pins:
# Set BOARD pin 32 as mux function PWM (func 1):
# sudo busybox devmem 0x2430040 32 0x401
# Set BOARD pin 33 as mux function PWM (func 2):
# sudo busybox devmem 0x2440020 32 0x402

# Jetson board pins assignment
_pin_nEN = 29
_pin_nFAULT = 31
_pin_D1PWM = 32
_pin_D2PWM = 33
_pin_D1DIR = 18
_pin_D2DIR = 22

class Diverters(object):
    MAX_SPEED = _max_speed

    def __init__(self):
        self.diverter1 = Motor(_pin_D1PWM, _pin_D1DIR)
        self.deverter2 = Motor(_pin_D2PWM, _pin_D2DIR)

        # gpio mode
        GPIO.setmode(GPIO.BOARD)

        # set _pin_nEN pin as an output pin with optional initial state of LOW
        # to enable drivers by default
        GPIO.setup(_pin_nEN, GPIO.OUT, initial=GPIO.LOW)
        
        # set _pin_nFAULT pin as an input pin
        GPIO.setup(_pin_nFAULT, GPIO.IN)  

    def setSpeeds(self, d1_speed, d2_speed):
        self.diverter1.setSpeed(d1_speed)
        self.deverter2.setSpeed(d2_speed)

    def getFault(self):
        return not GPIO.input(_pin_nFAULT)

    def enable(self):
        GPIO.output(_pin_nEN, 0)

    def disable(self):
        GPIO.output(_pin_nEN, 1)

    def forceStop(self):
        self.setSpeeds(0, 0)
        self.disable()
    
    def Cleanup(self):
        GPIO.cleanup()

