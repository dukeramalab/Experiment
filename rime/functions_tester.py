## This file sets up the data constructs for both control and data messages
## and then tests then for building and parsing.

from construct import *
import sys
import motemessage
import functions

# --------- test input parsing ---------

string = "aa01027a7a023c0061006161616101020304050607082d00610061616161404142434445464738000100020003000400050006000700080009000a000100020003000400050006000700080009000a0019001a000100020003000400050006000700080009000a001c06ed24"
mycontrol = motemessage.parse(string.decode("hex"))
mydata = motemessage.parse("5501020102030405060708404142434445464702050461736466".decode("hex"))

# --------- test output building ---------

mydatastring = motemessage.build(
    Container(
        message_type = 'DATA',
        FLAGS01 = 1,
        FLAGS02 = 2,
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
        FLAGS01 = 1,
        FLAGS02 = 2,
        message = Container(
            buffer_size = 122,
            present_buffer_used = 122,
            num_neighbors = 2,
            neighbor_table = [
                Container(
                    RSSI = 60,
                    decision_T = 97,
                    constraint1 = 24929,
                    lamada = 24929,
                    node_address = [1,2,3,4,5,6,7,8],
                ),
                Container(
                    RSSI = 45,
                    decision_T = 97,
                    constraint1 = 24929,
                    lamada = 24929,
                    node_address = [64,65,66,67,68,69,70,71],
                )
            ],
            eta = 56,
            uss = [1,2,3,4,5,6,7,8,9,10],
            slacks2 = [1,2,3,4,5,6,7,8,9,10],
            slacks = [25, 26],
            mu_cons2 = [1,2,3,4,5,6,7,8,9,10],
            mu_cons1 = 1564,
            mu = 9453,
        ),
    )
)

mycontrolstring2 = motemessage.build(
    Container(
        message_type = 'CONTROL',
        FLAGS01 = 1,
        FLAGS02 = 2,
        message = Container(
            buffer_size = 122,
            present_buffer_used = 122,
            num_neighbors = 2,
            neighbor_table = [
                Container(
                    RSSI = 60,
                    decision_T = 97,
                    constraint1 = 24929,
                    lamada = 24929,
                    node_address = [1,2,3,4,5,6,7,8],
                ),
                Container(
                    RSSI = 45,
                    decision_T = 97,
                    constraint1 = 24929,
                    lamada = 24929,
                    node_address = [64,65,66,67,68,69,70,71],
                )
            ],
            eta = 56,
            uss = [1,2,3,4,5,6,7,8,9,10],
            slacks2 = [1,2,3,4,5,6,7,8,9,10],
            slacks = [25, 26],
            mu_cons2 = [1,2,3,4,5,6,7,8,9,10],
            mu_cons1 = 1564,
            mu = 9453,
        ),
    )
)


# ---------------------
#functions.addpcmessage(mycontrol)
#functions.addpcmessage(mydata)

functions.testreadmessage(mycontrol)
print functions.getcontrolmessage('[97, 97, 97, 97, 97, 97, 97, 97]')
print functions.getcontrolall()
