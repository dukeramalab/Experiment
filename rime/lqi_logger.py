from construct import *
import serial
import sys
import motemessage

ser = serial.Serial("/dev/tty"+sys.argv[1], baudrate=115200, timeout=None)
ser.flush();

f = open('lqi.txt', 'w')

# --------- run ---------
while True:
    if (ser.isOpen()):
		# read serial		
		data = ser.readline()
		data = data.strip()
		# print data

		message_indicator = data[0:2]
		# decode and parse if it is a recognized message type
		if(message_indicator == "aa" or message_indicator == "55"):
			decoded_data = data.decode('hex')
			#print decoded_data		
			parsed_data = motemessage.parse(decoded_data)
			print str(parsed_data.message.neighbor_table[0].LQI)
			f.write(str(parsed_data.message.neighbor_table[0].LQI) + '\n')

ser.close()
