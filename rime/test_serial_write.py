## This file sets up the data constructs for both control and data messages
## and allows you to send them from PC to radio to test sending data to the radios

from construct import *
import sys
import motemessage
import moteio

# --------- build output ---------

mydatastring = motemessage.build(
    Container(
        message_type = 'DATA',
        part_number = 1,
        total_parts = 1,
        is_my_info = 0,
        message = Container(
            source_address = [1,2,3,4,5,6,7,8],
            destination_address = [64,65,66,67,68,69,70,71],
            sequence_number = 2,
            hop_count = 5,
            payload_length = 4,
            payload = "asdf",
        ),
    )
)

mycontrolstring = motemessage.build(
    Container(
        message_type = 'CONTROL',
        part_number = 1,
        total_parts = 4,
        is_my_info = 1,
        message = Container(
            RSSI = 55,
            node_address = [125,86,169,140,42,12,5,0],
            eta = 2,
            uss = [2,2,2,2,2,2,2,2,2,2],
            slacks2 = [2,2,2,2,2,2,2,2,2,2],
            slacks = [5, 5],
            mu_cons2 = [2,2,2,2,2,2,2,2,2,2],
            mu_cons1 = 5,
            mu = -30,
            decision_T = [5,5,5,5],
            Y_soln = [8,8,8,8],
        ),
    ),
)



# print mycontrolstring.encode('hex')
moteio.writeserial(mycontrolstring.encode('hex')+"\n", "/dev/"+sys.argv[1])
#moteio.writeserial(mycontrolstring2.encode('hex')+"\n", "/dev/"+sys.argv[1])
# moteio.writeserial(mycontrolstring.encode('hex')+"\n", "/dev/"+sys.argv[1])
# moteio.writeserial(mycontrolstring.encode('hex')+"\n", "/dev/"+sys.argv[1])
 
