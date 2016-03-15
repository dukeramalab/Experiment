## This file sets up the data constructs for both control and data messages
## and then reads data in from the serial port and parses it.

from construct import *
import serial
import sys
import motemessage

ser = serial.Serial("/dev/ttyUSB5", baudrate=115200, timeout=10)


# --------- run ---------
while True:
    if (ser.isOpen()):
        data = ser.readline()
        print data
        print motemessage.parse(data)
ser.close()
