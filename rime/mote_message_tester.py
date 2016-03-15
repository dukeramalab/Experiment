## This file sets up the data constructs for both control and data messages
## and then tests them for building and parsing.

from construct import *
import sys
import motemessage

# --------- test input parsing ---------

string = "aa010401370038000100020003000400050006000700080009000a000100020003000400050006000700080009000a0019001a000100020003000400050006000700080009000a001c06ed243c00460050005a0001000200030004000102030405060708"
mycontrol = motemessage.parse(string.decode("hex"))
mydata = motemessage.parse("550101000102030405060708404142434445464702050461736466".decode("hex"))

# --------- test output building ---------

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
            node_address = [1,2,3,4,5,6,7,8],
            eta = 56,
            uss = [1,2,3,4,5,6,7,8,9,10],
            slacks2 = [1,2,3,4,5,6,7,8,9,10],
            slacks = [25, 26],
            mu_cons2 = [1,2,3,4,5,6,7,8,9,10],
            mu_cons1 = 1564,
            mu = 9453,
            decision_T = [60,70,80,90],
            Y_soln = [1,2,3,4],
        ),
    ),
)

print mydata
print mycontrol
print mydatastring.encode("hex")
print mycontrolstring.encode("hex")
# prints the 3rd element of the node address
# print str(mycontrol.message.node_address[2])

##print mote_message.parse("\xaa\x00\x00\x1e\x04\x01\x7d\x82\x00\x00\x00\x00\x00\x00\x00\x70\x88\x40\x00\xa7\x4c\x40\x00\x00\x00\x00\x00\xf9\x4d\x40\x00\x30\x78\x40\x00\x08\x89\x40\x00\x08\x89\x40\x00\xd4\x89\x40\x00\xd4\x89\x40\x00\x9d\x3b\x40\x00\x98\x78\x40\x00\x82\x00\x0a\x00")

##while True:
##    data = raw_input()
##    data = data.strip()
##    print mote_control_message.parse(data)
    
##    for i in range(0, len(dataArray)):
##        if (dataArray[i][:3] == "LQI"):
##            storeLQI(dataArray[i][4:])
##        else:
##            parseData(dataArray[i])
