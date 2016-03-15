# Contains functions to read and write from the serial ports

from construct import *
import serial
import sys
import motemessage

# -------------------------------------------------------------
# functions
# -------------------------------------------------------------

# method to read data from serial port
# return Container structure of data that was read in
# call with serial port location, default is "/dev/ttyUSB1"
def readserial( portlocation="/dev/ttyUSB1" ):
    ser = serial.Serial(portlocation, baudrate=115200, timeout=10)
    ser.flush()
    if (ser.isOpen()):
        # read serial       
        data = ser.readline()
        data = data.strip()
        # decode and parse
        decoded_data = data.decode('hex')
        parsed_data = motemessage.parse(decoded_data)
    ser.close()
    # do something with date that has been read in
    return parsed_data

# method to write data to serial port
# call with serial port location, default is "/dev/ttyUSB1"
# call with Container which contains data to write to serial port
def writeserial( data, portlocation="/dev/ttyUSB1" ):
    ser = serial.Serial(portlocation, baudrate=115200, timeout=10)
    ser.flush()
    if (ser.isOpen()):
        # write to serial
        ser.write(data)
    ser.close()
