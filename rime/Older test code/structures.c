/* uint8_t message_type needs to be assigned before using this structure.
This acts as an identifier for the type of the structure that needs to be used for 
storage. 
message_type = 0xAA for Control Message
message_type = 0x55 fr Data Message

/* This is the structure for communicating control messages from 
the PC to mote, PC to mote, mote to mote*/

struct mote_control_message{
	uint8_t message_type; // Start byte used to indicate the type of Message. It is 0xAA for data message
	uint16_t FLAGS01; // Flags for future purpose
	uint8_t buffer_size; 
	uint8_t present_buffer_used;
	uint8_t num_neighbors; // Number of neighbors. Should be set when using the structure
	struct neighbor_table{ 
		uint8_t node_address[8];		
		uint16_t LQI;
		uint16_t RSSI;
		/* Percentage measure of the packets that need to routed to respective node (PC->mote)
		Percentage measure of the packets that are being routed from the jth node to 
		this node(ith node) (mote->PC)*/
		uint8_t decision_T; 

		/*Constraint required for optimization (ith -> jth ) (PC->mote)
		Constraint required for optimization (jth -> ith ) (mote->PC)*/
		uint16_t constraint1;

		/*Dual variable required for optimization (ith -> jth ) (PC->mote)
		Dual variable required for optimization (jth -> ith ) (mote->PC)*/
		uint16_t lamada;
		
		} neighbor_params[MAX_NEIGHBORS];
	char message_end;//message should not contain "\n" or null character
	
};


/* This is the structure for communicating Data messages from 
the PC to mote, PC to mote, mote to mote*/ 
struct mote_data_message{
	uint8_t message_type; // Start byte used to indicate the type of Message. It is 0x55 for data message
	uint16_t FLAGS01; // Flags for future purpose
	unsigned char source_address[8];
	unsigned char destination_address[8];
	uint8_t sequence_number; 
	uint8_t hop_count;
	uint8_t decision_T; // Percentage measure of the packets that are being routed to the destination node
	uint16_t constraint1; // Constraint required for optimization
	uint16_t lamada; // Dual variable
	uint8_t payload_length; // Payload length Let us Fix it to 32 for initial testing purposes
	char 	payload[DATA_PAYLOAD_LENGTH]; // Actual payload **Important Payload should not contain '\0' or null character
	char message_end;
};