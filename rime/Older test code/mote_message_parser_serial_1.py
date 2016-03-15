from construct import *
import binascii
import serial
import sys

ser = serial.Serial("/dev/ttyUSB1", baudrate=115200, timeout=10)
ser.flush()


neighbor_table = Struct("neighbor_table",
    Array(8, UBInt8("node_address")),	
    UBInt16("LQI"),
    UBInt16("RSSI"),
    UBInt16("constraint1"),
    UBInt16("lamada"),
	UBInt8("decision_T"),
)

mote_control_message = Struct("mote_control_message",    
	UBInt8("FLAGS01"),
	UBInt8("FLAGS02"),
    UBInt8("buffer_size"),
    UBInt8("present_buffer_used"),
    UBInt8("num_neighbors"),
##    Array(lambda ctx : ctx.num_neighbors, neighbor_table),
    Array(3, neighbor_table),
)


mote_data_message = Struct("mote_data_message",
    UBInt16("FLAGS01"),
    Array(8, UBInt8("source_address")),
    Array(8, UBInt8("destination_address")),
    UBInt8("sequence_number"),
    UBInt8("hop_count"),
    UBInt8("decision_T"),
    UBInt16("constraint1"),
    UBInt16("lamada"),
    UBInt8("payload_length"),
    String("payload", lambda ctx : ctx.payload_length), # may not want to read specific amount if want to error check
)

mote_message = Struct("mote_message",
	UBInt8("Message_type"),
    UBInt8("FLAGS01"),
	UBInt8("FLAGS02"),
    UBInt8("buffer_size"),
    UBInt8("present_buffer_used"),
    UBInt8("num_neighbors"),
##    Array(lambda ctx : ctx.num_neighbors, neighbor_table),
    Array(3, neighbor_table),
    #String("message_end", 1),
    #RepeatUntil(lambda obj, ctx: obj == "\n", String("message_end", 1)),
	String("message_end",1),
)


# --------- run ---------
x = 1
while x:
    if (ser.isOpen()):
		#ser.flush()
		#sw = ser.read()+ser.read()+ser.read()+ser.read()
		#print sw
		#if(sw == "\xaa" or sw == '\x55'):        
			#data = ser.readline()
			#print data
			#print mote_message.parse(data)
		#else:			
		#data = mote_message.unpack("<H",ser.readline())
		#data = "\xaa\x00\x9c\x1e\x00\x01\x7f\x4c\x40\x00\x01\x00\x00\x00\xa7\x4d\x40\x00\x00\x00\x00\x00\x00\x00\x00\x00\x9c\xab\x40\x00\x55\x4b\x40\x00\xc0\x8e\x40\x00\x88\x00\x00\x00\x78\x7d\x40\x00\xad\x3c\x40\x00\xa8\x8f\x40\x00\x29\x3c\x40\x0a"
		#data= data.strip('\x')
		data = ser.readline()
		data = data.strip()	
		print data
		#data_2=mote_message.parse(data)
		#print data_2
		#data = "aa00001e00018ca9567d8200000000000000e88740002f5240000000000081534000c08e400088000000787d4000ad3c4000a88f4000293c400a"

		print len(data)
		decoded_data = data.decode('hex')
		print decoded_data		
		parsed_data = mote_message.parse(decoded_data)
		#parsed_data= binascii.unhexlify(data)
		print parsed_data
		
		#built_data = mote_message.build(parsed_data)
		#print built_data
		#print str(data)
		#print mote_message.parse(str(data))
		#print mote_message.parse("\xaa\x00\x9c\x1e\x00\x01\x7f\x4c\x40\x00\x01\x00\x00\x00\xa7\x4d\x40\x00\x00\x00\x00\x00\x00\x00\x00\x00\x9c\xab\x40\x00\x55\x4b\x40\x00\xc0\x8e\x40\x00\x88\x00\x00\x00\x78\x7d\x40\x00\xad\x3c\x40\x00\xa8\x8f\x40\x00\x29\x3c\x40\x0a")
    
ser.close()
