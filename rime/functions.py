## Includes functions to read/write data/control messages to and from the
## serial ports and store them into the appropriate data structures.

from construct import *
import serial
import sys
import motemessage
import moteio

controlmap = {}
datamap = {}


# -------------------------------------------------------------
# callable functions
# -------------------------------------------------------------

def testreadmessage( data ):
    if (data.message_type == 'DATA'):
        datamap[str(data.message.source_address[0:8])] = data
    if (data.message_type == 'CONTROL'):
        listofneighbors = data.message.neighbor_table
        for n in listofneighbors:
            controlmap[str(data.message.neighbor_table[0].node_address[0:8])] = n
    
def readdatamessage( portlocation="/dev/ttyUSB1" ):
    moteio.writeserial("aa", portlocation)
    readcomplete = 0
    data = Container()
    while ( readcomplete == 0 ):
        data = moteio.readserial(portlocation)
        if ( data.message_type == 'DATA' ):
            readcomplete = 1
    datamap[str(data.message.source_address[0:8])] = data

def readcontrolmessage( portlocation="/dev/ttyUSB1" ):
    moteio.writeserial("55", portlocation)
    readcomplete = 0
    data = Container()
    while ( readcomplete == 0 ):
        data = moteio.readserial(portlocation)
        if ( data.message_type == 'CONTROL' ):
            readcomplete = 1
    listofneighbors = data.message.neighbor_table
    for n in listofneighbors:
        controlmap[str(data.message.neighbor_table[0].node_address[0:8])] = n

def getcontrolall():
    return controlmap

def getcontrolmessage( neighbor ):
    try:
        return controlmap[neighbor]
    except KeyError:
        return 'No neighbor with address ' + neighbor + ' found.'

def getdatamessage( neighbor ):
    try:
        return datamap[neighbor]
    except KeyError:
        return 'No neighbor with address ' + neighbor + ' found.'
