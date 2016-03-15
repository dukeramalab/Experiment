## This file allows you to enter one encoded string and parses the result into readable parameters

from construct import *
import sys
import motemessage

# change this line to contain your message you want to parse
data = "aa00001e0001e10000000000140000007d56a98c2a0c050000b74000eb50400000000000135240000000000000000000249040008800000001000000753f4000249040009b3f4000f4994000f13e0d00"

# parse and print the message
data= data.strip()
print data
message_indicator = data[0:2]
# decode and parse if it is a recognized message type
if(message_indicator == "aa" or message_indicator == "55"):
    decoded_data = data.decode('hex')
    #print decoded_data     
    parsed_data = motemessage.parse(decoded_data)
    print parsed_data
