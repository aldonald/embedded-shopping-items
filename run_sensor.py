import RPi.GPIO as GPIO
import time
from shopping_list_camera import run_camera 


GPIO.setmode(GPIO.BOARD)

#define the pin that goes to the circuit
pin = 8

GPIO.setup(pin, GPIO.IN)

try:
    while True:
        sensor = GPIO.input(pin)
        if sensor == 0:
            time.sleep(0.5)
        elif sensor == 1:
            print("Movement detected")
            run_camera()
            time.sleep(10)

except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()

