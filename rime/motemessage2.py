from construct import *

NUM_SCENARIOS = 10
NUM_NODES = 4

# -------------------------------------------------------------
# define data structure
# -------------------------------------------------------------

# neighbor_info contains all the data for a single neighbor, 96 bytes
neighbor_info = Struct("neighbor_info",
    ULInt16("RSSI"),
    SLInt16("eta"),
    Array(NUM_SCENARIOS, ULInt16("uss")),
    Array(NUM_SCENARIOS, ULInt16("slacks2")),
    Array(2, ULInt16("slacks")),
    Array(NUM_SCENARIOS, SLInt16("mu_cons2")),
    SLInt16("mu"),
    SLInt16("mu_cons1"),
    Array(NUM_NODES, ULInt16("decision_T")),
    Array(NUM_NODES, ULInt16("Y_soln")),
    Array(8, ULInt8("node_address")),
)

# stores payload data to send between motes
mote_data_message = Struct("mote_data_message",
    Array(8, ULInt8("source_address")),
    Array(8, ULInt8("destination_address")),
    ULInt8("sequence_number"),
    ULInt8("hop_count"),
    ULInt8("payload_length"),
    String("payload", lambda ctx : ctx.payload_length), # may not want to read specific amount if want to error check
)

# this is the header for all note messages, 
mote_message = Struct("mote_message",
    Enum(ULInt8("message_type"),
         CONTROL = 0xAA,
         DATA = 0x55,
    ),
    ULInt8("part_number"),
    ULInt8("total_parts"),
    ULInt8("is_my_info"),
    Switch("message", lambda ctx: ctx.message_type,
        {
            "CONTROL" : neighbor_info,
            "DATA" : mote_data_message,
        },
        default = Pass
    ),
)

# -------------------------------------------------------------
# functions
# -------------------------------------------------------------
def parse( data ):
    return mote_message.parse(data)

def build( data ):
    return mote_message.build(data)
