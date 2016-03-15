#!/usr/bin/env python

# to test, use: python store_lqi_grid.py < lqitest.in
# in order to redirect test file as input

# take in data and LQI values
# parse data and store LQI values into grid to send out

# expects input stream to be space-delimited and for any LQI values
# to be in the format LQI.aa.xxx where a and x are both integer values
# a = id of node from which data came from, value 0-99
# x = LQI value 0-255

from construct import *
import array
#import serial
import sys

#ser = serial.Serial("/dev/ttyUSB1", baudrate=115200, timeout=10)

lqiArray = array.array('i',(0 for i in range(0,10)))
LQIstring = Struct("LQIstring",
    UBInt8("length"),
    Bytes("data", lambda ctx: ctx.length),)
LQIpacket = ExprAdapter(LQIstring,
    encoder = lambda obj, ctx: Container(length = len(obj), data = obj),
    decoder = lambda obj, ctx: obj.data)

def storeLQI(lqi):
    value = lqi.split('.')
    lqiArray[int(value[0])] = int(value[1])
    print lqiArray

def parseData(d):
    print d


#while True:
for data in sys.stdin:
    #if (ser.isOpen()):
        #data = ser.read()
        #data = raw_input()
        
        data = data.strip()
        dataArray = data.split(' ')
        for i in range(0, len(dataArray)):
            if (dataArray[i][:3] == "LQI"):
                storeLQI(dataArray[i][4:])
            else:
                parseData(dataArray[i])
                
#ser.close()
