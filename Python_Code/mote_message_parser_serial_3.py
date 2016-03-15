from construct import *
import serial
import sys
import motemessage2
import numpy as np
import ADAL
import time
import moteio

ser = serial.Serial("/dev/"+sys.argv[1], baudrate=115200, timeout=None)
# ser = serial.Serial("/dev/ttyUSB5", baudrate=115200, timeout=10)

ser.flush();
count = 0
mycontrolstring_right = motemessage2.build(
    Container(
        message_type = 'CONTROL',
        part_number = 1,
        total_parts = 4,
        is_my_info = 1,
        message = Container(
            RSSI = 55,
            node_address = [107,105,118,140,42,12,5,0],
            eta = 560,
            uss = [1000,2000,3000,4000,5000,5000,5000,5000,5000,1000],
            slacks2 = [1000,2000,3000,4000,5000,5000,5000,5000,5000,1000],
            slacks = [25000, 26000],
            mu_cons2 = [-100,-100,-100,-100,-100,-100,-200,-300,-400,-500],
            mu_cons1 = 156,
            mu = 545,
            decision_T = [6000,5000,5000,5000],
            Y_soln = [9,9,9,9],
        ),
    ),
) 

mycontrolstring_left = motemessage2.build(
    Container(
        message_type = 'CONTROL',
        part_number = 1,
        total_parts = 4,
        is_my_info = 1,
        message = Container(
            RSSI = 55,
            node_address = [125,86,169,140,42,12,5,0],
            eta = 56,
            uss = [1000,2000,3000,4000,5000,5000,5000,5000,5000,1000],
            slacks2 = [1000,2000,3000,4000,5000,5000,5000,5000,5000,1000],
            slacks = [2500, 2600],
            mu_cons2 = [-100,-100,-100,-100,-100,-100,-200,-300,-400,-500],
            mu_cons1 = 156,
            mu = 545, 
            decision_T = [6000,5000,5000,5000],
            Y_soln = [9,9,9,9],
        ),
    ),
) 

mycontrolstring_back = motemessage2.build(
    Container(
        message_type = 'CONTROL',
        part_number = 1,
        total_parts = 4,
        is_my_info = 1,
        message = Container(
            RSSI = 55,
            node_address = [165,64,240,140,42,12,5,0],
            eta = 56,
            uss = [1000,2000,3000,4000,5000,5000,5000,5000,5000,1000],
            slacks2 = [1000,2000,3000,4000,5000,5000,5000,5000,5000,1000],
            slacks = [2500, 2600],
            mu_cons2 = [-100,-100,-100,-100,-100,-100,-200,-300,-400,-500],
            mu_cons1 = 156,
            mu = 545, 
            decision_T = [6000,5000,5000,5000],
            Y_soln = [1,1,1,1],
        ),
    ),
) 
# A is a dictionary to store data from neighbors including myself
A = {}

# --------- run ---------
while True:
    if (ser.isOpen()):
        # read serial	
        #for counter in range(0, 2): 	
        data = ser.readline()
        data = data.strip()
        print data,

        message_indicator = data[0:2]
        # decode and parse if it is a recognized message type
        if(message_indicator == "aa" or message_indicator == "55"):
            decoded_data = data.decode('hex')
            # print decoded_data		
            parsed_data = motemessage2.parse(decoded_data)
            print parsed_data

            # if parsed_data.message.node_address[0] == 125:
            #     A['Node3'] = parsed_data
            #     print str(A['Node3'].message.node_address[0])
            #     input_3 = np.asarray( A['Node3'].message.decision_T[0:2] )
            #     print input_3


            # Mapping: node_address -> position in neighbor list
            # We can use node address to build a global mapping between node_address and node
            # if parsed_data.message.node_address[0] == 125:
            #     A['Node1'] = parsed_data
            #     print str(A['Node1'].message.node_address[0])
            #     input_1 = np.asarray( A['Node1'].message.decision_T )

            # elif parsed_data.message.node_address[0] == 107:
            #     A['Node2'] = parsed_data
            #     print str(A['Node2'].message.node_address[0])
            #     input_2 = np.asarray( A['Node2'].message.decision_T )

            # elif parsed_data.message.node_address[0] == 165:
            #     A['Node3'] = parsed_data  
            #     print str(A['Node3'].message.node_address[0])
            #     input_3 = np.asarray( A['Node3'].message.decision_T )

            # elif parsed_data.message.node_address[0] == 222:
            #     A['Node4'] = parsed_data
            #     print str(A['Node4'].message.node_address[0])   
            #     input_4 = np.asarray( A['Node4'].message.decision_T )   
             
            # Call algorithm
            # time.sleep(5)
            # [decision_T] = ADAL.algorithm(input_1,input_2,input_3,input_4)
            
            # num_neighbors = 2
            # if  len( A ) == num_neighbors:
            #     [decision_T] = ADAL.algorithm(input_1,input_2)
            #     decision_T = decision_T.tolist()
            #     # print decision_T
            #     #A['Node1'].message.decision_T = decision_T
            #     A['Node1'].message.mu = 335
            #     parsed_data = A['Node1']

        # time.sleep(.01)
        # parsed_data = A['Node2']
        # moteio.writeserial(mycontrolstring.encode('hex')+"\n", "/dev/"+sys.argv[1])
        # moteio.writeserial(motemessage.build(parsed_data).encode('hex')+"\n", "/dev/"+sys.argv[1])
        # time.sleep(.01)

        # count = count + 1
        # if count > 10:
        #     ser.flush()
        #     if (ser.isOpen()):
        #         ser.write(mycontrolstring_right.encode('hex')+"\n")
        #         #ser.write(mycontrolstring_left.encode('hex')+"\n")
        #         #ser.write(mycontrolstring_back.encode('hex')+"\n")
        #         print "Wrote to serial port"
        #         count = 0
ser.close()
