import cv2
import time
import pyttsx3
import RPi.GPIO as GPIO
from object_detection import *

THRESHOLD = 100
GPIO.setmode(GPIO.BCM)

GPIO_TRIGGER = 18
GPIO_ECHO = 24
GPIO_BUZZER = 17

GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)
GPIO.setup(GPIO_BUZZER, GPIO.OUT)


def calculate_distance():
    GPIO.output(GPIO_TRIGGER, True)
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)

    start_time = time.time()
    stop_time = time.time()

    while GPIO.input(GPIO_ECHO) == 0:
        start_time = time.time()

    while GPIO.input(GPIO_ECHO) == 1:
        stop_time = time.time()

    time_elapsed = stop_time - start_time
    distance = (time_elapsed * 34300) / 2

    return distance


if __name__ == '__main__':
    engine = pyttsx3.init()
    vid_capture = cv2.VideoCapture(0)
    time.sleep(2)
    try:
        engine.startLoop(False)
        while True:
            engine.iterate()
            dist = calculate_distance()
            print(f"Measured Distance = {dist} cm")
            if dist >= THRESHOLD:
                print('Person Got Too Close')
                GPIO.output(GPIO_BUZZER, True)
                ret, frame = vid_capture.read()
                detect_objects_and_speak(frame, dist)
                time.sleep(4)
            else:
                print('Person Is Safe Now')
                GPIO.output(GPIO_BUZZER, False)
            time.sleep(1)
    except Exception as e:
        print(e)
        print("Measurement Stopped By User")
        GPIO.cleanup()
        engine.endLoop()
