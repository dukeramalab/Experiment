from construct import *
import numpy
from cvxpy import *
import serial
import sys
import motemessage

ser = serial.Serial("/dev/ttyUSB1", baudrate=115200, timeout=10)
ser.flush()

local_lamada = 0 
local_constraint = 0

print 
# --------- run ---------
while True:
    if (ser.isOpen()):
		# read serial		
		data = ser.readline()
		data= data.strip()
		#print data
		message_indicator = data[0:2]
		# decode and parse if it is a recognized message type
		if(message_indicator == "aa" or message_indicator == "55"):
			decoded_data = data.decode('hex')
			#print decoded_data		
			parsed_data = motemessage.parse(decoded_data)
			#print parsed_data

			#Number of Neighbors N_i
			N_i = parsed_data.message.num_neighbors
			lamada_array = numpy.zeros((N_i+1,1))
			constraint1_array = numpy.zeros((N_i+1,1))
			R_i_array = numpy.zeros((N_i+1,N_i+1))
			for j in range(N_i):
				#print parsed_data.message.neighbor_table[i].lamada			
				lamada_array[j] = float(parsed_data.message.neighbor_table[j].lamada)
				R_i_array[j,j+1] = -parsed_data.message.neighbor_table[j].LQI/256
				constraint1_array[j] = float(parsed_data.message.neighbor_table[j].constraint1)
				R_i_array[-1,j+1] = parsed_data.message.neighbor_table[j].LQI/256
			lamada_array[-1]= local_lamada
			R_i_array[-1,1] = 1
			constraint1_array[-1] = local_constraint
			
			x = Variable90
			print lamada_array
		
   
ser.close()
