from construct import *
import serial
import sys
import motemessage2
import time
import moteio

from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import numpy as np
import sink5
import sink5_dual
import math

dummy = Container(
        message_type = 'CONTROL',
        part_number = 0,
        total_parts = 0,
        is_my_info = 0,
        message = Container(
            RSSI = 0,
            node_address = [165,64,240,140,42,12,5,0],
            eta = 0,
            uss = [0,0,0,0,0,0,0,0,0,0],
            slacks2 = [0,0,0,0,0,0,0,0,0,0],
            slacks = [0, 0],
            mu_cons2 = [0,0,0,0,0,0,0,0,0,0],
            mu_cons1 = 0,
            mu = 0, 
            decision_T = [0,0,0,0],
            Y_soln = [0,0,0,0],
        )
    ) 

def make2dList(rows, cols):
    a=[]
    for row in xrange(rows): a += [[0]*cols]
    return a

ser = serial.Serial("/dev/"+sys.argv[1], baudrate=115200, timeout=None)
# ser = serial.Serial("/dev/ttyUSB5", baudrate=115200, timeout=10)

ser.flush();
count = 0

# Initialization 
Ns = 10
J = 2
K = 1
lamba = 0.0
beta = 0.9
max_iter_adal = 500
rho = 10
epsilon = 0.0
norm_option = 1

r_min_r = np.random.rand(J,1)
r_min_r = 0.2*np.ones(J).reshape(J,1)

R_r = []
R_r_indi = [] 
for v in range(Ns):
    # R_r.append( np.random.rand(J,J+K) ) 
    R_r.append( 0.5*np.ones(J*(J+K)).reshape(J,J+K) )

    R_r[v][0,0] = 0
    R_r[v][1,1] = 0
    #R_r[v][1,0] = 0
     
    R_r_indi.append( np.zeros((J,J+K)) )

R_mean = np.random.rand(J,J+K)
R_mean = 0.5*np.ones(J*(J+K)).reshape(J,J+K)

R_mean[0,0] = 0
R_mean[1,1] = 0
#R_mean[1,0] = 0     

rows = J
cols = Ns
RR_r = make2dList(rows, cols) 

rows = J
cols = 1
RR_mean = make2dList(rows, cols) 
    
for scen in range(Ns):    
    for i in range(J):
        colvect = 0*np.ones(J).reshape(J,1)
        colvect[i,0] = 1
        RR_2 = np.diagflat(R_r[scen][i,0:J])
        RR_r[i][scen] = np.kron( colvect, -R_r[scen][i,0:J].reshape(1,J) ) + RR_2

for i in range(J):
    colvect = 0*np.ones(J).reshape(J,1)
    colvect[i,0] = 1
    RR_2_mean = np.diagflat(R_mean[i,0:J])
    RR_mean[i][0] = np.kron( colvect, -R_mean[i,0:J].reshape(1,J) ) + RR_2_mean        

for v in range(Ns):
    for i in range(J):
        for j in range(J+K):
            if R_r[v][i,j] > 0.001:
                R_r_indi[v][i,j] = 1.0

R_r_indi_2 = np.zeros((J,Ns))                 
for v in range(Ns):
    for i in range(J):
        R_r_indi_2[i,v] = np.sum( R_r_indi[v][i,:], 0 )
        
tau = 1/np.amax( R_r_indi_2 ) 

T_n = []
etaa = []
uss = []
slacks_2 = []
slacks = []
for i in range(J):
    T_n.append( np.zeros((J,max_iter_adal+1)) )
    etaa.append( np.zeros((1,max_iter_adal+1)) )
    uss.append( np.zeros((Ns,max_iter_adal+1)) )
    slacks_2.append( np.zeros((Ns,max_iter_adal+1)) )
    slacks.append( np.zeros((2,max_iter_adal+1)) )

T_s = [] 
Y_soln = []
for j in range(K):
    T_s.append( np.zeros((J,max_iter_adal+1)) )
    Y_soln.append( np.zeros((J,max_iter_adal+1)) )

mu_cons2 = np.zeros((J*Ns,max_iter_adal+1))  
mu_cons1 = np.zeros((J,max_iter_adal+1))
mu = np.zeros((J,max_iter_adal+1))

constraints_2 = np.zeros((J*Ns,max_iter_adal)) 
constraints_1 = np.zeros((J,max_iter_adal)) 
sum_constraints = np.zeros((J,max_iter_adal))

max_constraints_2 = np.zeros((1,max_iter_adal)) 
max_constraints_1 = np.zeros((1,max_iter_adal)) 
max_sum_constraints = np.zeros((1,max_iter_adal))

obj_cvar_adal = np.zeros((1,max_iter_adal))    

input_dic = {}
input_dic['T_n'] = T_n 
input_dic['etaa'] = etaa  
input_dic['uss'] = uss
input_dic['slacks_2'] = slacks_2   
input_dic['slacks'] = slacks   
input_dic['T_s'] = T_s  
input_dic['Y_soln'] = Y_soln 
input_dic['mu_cons2'] = mu_cons2   
input_dic['mu_cons1'] = mu_cons1
input_dic['mu'] = mu
input_dic['constraints_2'] = constraints_2   
input_dic['constraints_1'] = constraints_1
input_dic['sum_constraints'] = sum_constraints
input_dic['max_constraints_2'] = max_constraints_2   
input_dic['max_constraints_1'] = max_constraints_1
input_dic['max_sum_constraints'] = max_sum_constraints
input_dic['obj_cvar_adal'] = obj_cvar_adal

# ULInt16("RSSI"),
# SLInt16("eta"),
# Array(NUM_SCENARIOS, ULInt16("uss")),
# Array(NUM_SCENARIOS, ULInt16("slacks2")),
# Array(2, ULInt16("slacks")),
# Array(NUM_SCENARIOS, SLInt16("mu_cons2")),
# SLInt16("mu_cons1"),
# SLInt16("mu"),
# Array(NUM_NODES, ULInt16("decision_T")),
# Array(NUM_NODES, ULInt16("Y_soln")),
# Array(8, ULInt8("node_address")),

# A is a dictionary to store data from neighbors including its own data
A = {}

node_id = 0

opt_iter = 0

# --------- run ---------
while True:
    if (ser.isOpen()):

            # read serial	 	
            data = ser.readline()
            data = data.strip()
            print data,

            message_indicator = data[0:2]
            # decode and parse if it is a recognized message type
            if(message_indicator == "aa" or message_indicator == "55"):
                decoded_data = data.decode('hex')
                # print decoded_data		
                parsed_data = motemessage2.parse(decoded_data)
                # print parsed_data

                # Mapping: node_address -> position in neighbor list
                if parsed_data.message.node_address[0] == 107:
                    A['Node1'] = parsed_data ## Source 1
                    print str(A['Node1'].message.node_address[0])
                    input_dic['T_n'][0][:,opt_iter] = np.asarray( A['Node1'].message.decision_T[0:2] )
                    input_dic['etaa'][0][:,opt_iter] = np.asarray( A['Node1'].message.eta )
                    input_dic['uss'][0][:,opt_iter] = np.asarray( A['Node1'].message.uss )
                    input_dic['slacks_2'][0][:,opt_iter] = np.asarray( A['Node1'].message.slacks2 )
                    input_dic['slacks'][0][:,opt_iter] = np.asarray( A['Node1'].message.slacks )
                    # input_dic['Y_soln'][0][:,opt_iter] = np.asarray( A['Node1'].message.Y_soln[0:2] )
                    input_dic['mu_cons2'][0:10,opt_iter] = np.asarray( A['Node1'].message.mu_cons2 )
                    input_dic['mu_cons1'][0:1:,opt_iter] = np.asarray( A['Node1'].message.mu_cons1 )
                    input_dic['mu'][0:1,opt_iter] = np.asarray( A['Node1'].message.mu )

                elif parsed_data.message.node_address[0] == 125:
                    A['Node3'] = parsed_data ## Source 2
                    print str(A['Node3'].message.node_address[0])
                    input_dic['T_n'][1][:,opt_iter] = np.asarray( A['Node3'].message.decision_T[0:2] )
                    input_dic['etaa'][1][:,opt_iter] = np.asarray( A['Node3'].message.eta )
                    input_dic['uss'][1][:,opt_iter] = np.asarray( A['Node3'].message.uss )
                    input_dic['slacks_2'][1][:,opt_iter] = np.asarray( A['Node3'].message.slacks2 )
                    input_dic['slacks'][1][:,opt_iter] = np.asarray( A['Node3'].message.slacks )
                    # input_dic['Y_soln'][1][:,opt_iter] = np.asarray( A['Node3'].message.Y_soln[0:2] )
                    input_dic['mu_cons2'][10:20,opt_iter] = np.asarray( A['Node3'].message.mu_cons2 )
                    input_dic['mu_cons1'][1:2,opt_iter] = np.asarray( A['Node3'].message.mu_cons1 )
                    input_dic['mu'][1:2,opt_iter] = np.asarray( A['Node3'].message.mu ) 

                elif parsed_data.message.node_address[0] == 165:
                    A['Node5'] = parsed_data ## Sink 1  
                    print str(A['Node5'].message.node_address[0])
                    input_dic['T_s'][0][:,opt_iter] = np.asarray( A['Node5'].message.decision_T[0:2] )
                    # input_dic['etaa'][2][:,opt_iter] = np.asarray( A['Node5'].message.eta )
                    # input_dic['uss'][2][:,opt_iter] = np.asarray( A['Node5'].message.uss )
                    # input_dic['slacks_2'][2][:,opt_iter] = np.asarray( A['Node5'].message.slacks2 )
                    # input_dic['slacks'][2][:,opt_iter] = np.asarray( A['Node5'].message.slacks )
                    input_dic['Y_soln'][0][:,opt_iter] = np.asarray( A['Node5'].message.Y_soln[0:2] )
                    # input_dic['mu_cons2'][20:30,opt_iter] = np.asarray( A['Node5'].message.mu_cons2 )
                    # input_dic['mu_cons1'][2:3,opt_iter] = np.asarray( A['Node5'].message.mu_cons1 )
                    # input_dic['mu'][2:3,opt_iter] = np.asarray( A['Node5'].message.mu ) 

                count = count + 1

                # print opt_iter
                # # print A['Node1'].message.decision_T[0:2]
                # print input_dic['T_n'][0][:,opt_iter] 
                # print input_dic['T_n'][1][:,opt_iter]
                # print input_dic['T_s'][0][:,opt_iter] 

                # Call primal algorithm
                # if primal condition happens:
                # if ( opt_iter  == 0 )  or ( input_dic['mu'][0:1,opt_iter]*math.pow(10,12) != 0 and input_dic['mu'][1:2,opt_iter]*math.pow(10,12) != 0 ):
                if  count == 5: 

                    input_dic['T_n'][0][:,opt_iter] = input_dic['T_n'][0][:,opt_iter].astype(float)/1000  
                    input_dic['etaa'][0][:,opt_iter] = input_dic['etaa'][0][:,opt_iter].astype(float)/1000  
                    input_dic['uss'][0][:,opt_iter] = input_dic['uss'][0][:,opt_iter].astype(float)/1000  
                    input_dic['slacks_2'][0][:,opt_iter] = input_dic['slacks_2'][0][:,opt_iter].astype(float)/1000  
                    input_dic['slacks'][0][:,opt_iter] = input_dic['slacks'][0][:,opt_iter].astype(float)/1000  
                    # input_dic['Y_soln'][0][:,opt_iter] =   
                    input_dic['mu_cons2'][0:10,opt_iter] = input_dic['mu_cons2'][0:10,opt_iter].astype(float)/100  
                    input_dic['mu_cons1'][0:1:,opt_iter] = input_dic['mu_cons1'][0:1:,opt_iter].astype(float)/100  
                    input_dic['mu'][0:1,opt_iter] = input_dic['mu'][0:1,opt_iter].astype(float)/100 

                    input_dic['T_n'][1][:,opt_iter] = input_dic['T_n'][1][:,opt_iter].astype(float)/1000  
                    input_dic['etaa'][1][:,opt_iter] = input_dic['etaa'][1][:,opt_iter].astype(float)/1000  
                    input_dic['uss'][1][:,opt_iter] = input_dic['uss'][1][:,opt_iter].astype(float)/1000  
                    input_dic['slacks_2'][1][:,opt_iter] = input_dic['slacks_2'][1][:,opt_iter].astype(float)/1000  
                    input_dic['slacks'][1][:,opt_iter] = input_dic['slacks'][1][:,opt_iter].astype(float)/1000  
                    # input_dic['Y_soln'][1][:,opt_iter] =   
                    input_dic['mu_cons2'][10:20,opt_iter] = input_dic['mu_cons2'][10:20,opt_iter].astype(float)/100  
                    input_dic['mu_cons1'][1:2,opt_iter] = input_dic['mu_cons1'][1:2,opt_iter].astype(float)/100  
                    input_dic['mu'][1:2,opt_iter] = input_dic['mu'][1:2,opt_iter].astype(float)/100 

                    input_dic['T_s'][0][:,opt_iter] = input_dic['T_s'][0][:,opt_iter].astype(float)/1000 
                    input_dic['Y_soln'][0][:,opt_iter] = input_dic['Y_soln'][0][:,opt_iter].astype(float)/1000  
                     
                    output_dic = sink5.algorithm(node_id, opt_iter, input_dic)

                    input_dic['T_s'][0][:,opt_iter+1] = ( output_dic['T_s'][0][:,opt_iter+1]*1000 ).astype(int)  
                    input_dic['Y_soln'][0][:,opt_iter+1] = ( output_dic['Y_soln'][0][:,opt_iter+1]*1000 ).astype(int)  
                      
                    A['Node5'] = dummy
                    A['Node5'].message.decision_T[0:2] = input_dic['T_s'][0][:,opt_iter+1].astype(int).tolist()   
                    A['Node5'].message.Y_soln[0:2] = input_dic['Y_soln'][0][:,opt_iter+1].astype(int).tolist()   
                       
                    parsed_data = A['Node5']
                    print parsed_data

                    ser.flush()
                    if (ser.isOpen()):
                        ser.write(motemessage2.build(parsed_data).encode('hex')+"\n")
                        print "Wrote to serial port ( primal update: sink 5 )"
                    opt_iter = opt_iter + 1

                # Call dual algorithm
                # if dual condition happens:
                # if  ( input_dic['etaa'][0][:,opt_iter]*math.pow(10,12) != 0 ) and ( input_dic['etaa'][1][:,opt_iter]*math.pow(10,12) != 0 ):  
                if count == 10:

                    input_dic['T_n'][0][:,opt_iter] = input_dic['T_n'][0][:,opt_iter].astype(float)/1000  
                    input_dic['etaa'][0][:,opt_iter] = input_dic['etaa'][0][:,opt_iter].astype(float)/1000  
                    input_dic['uss'][0][:,opt_iter] = input_dic['uss'][0][:,opt_iter].astype(float)/1000  
                    input_dic['slacks_2'][0][:,opt_iter] = input_dic['slacks_2'][0][:,opt_iter].astype(float)/1000  
                    input_dic['slacks'][0][:,opt_iter] = input_dic['slacks'][0][:,opt_iter].astype(float)/1000  
                    # input_dic['Y_soln'][0][:,opt_iter] =   
                    input_dic['mu_cons2'][0:10,opt_iter] = input_dic['mu_cons2'][0:10,opt_iter].astype(float)/100  
                    input_dic['mu_cons1'][0:1:,opt_iter] = input_dic['mu_cons1'][0:1:,opt_iter].astype(float)/100  
                    input_dic['mu'][0:1,opt_iter] = input_dic['mu'][0:1,opt_iter].astype(float)/100 

                    input_dic['T_n'][1][:,opt_iter] = input_dic['T_n'][1][:,opt_iter].astype(float)/1000  
                    input_dic['etaa'][1][:,opt_iter] = input_dic['etaa'][1][:,opt_iter].astype(float)/1000  
                    input_dic['uss'][1][:,opt_iter] = input_dic['uss'][1][:,opt_iter].astype(float)/1000  
                    input_dic['slacks_2'][1][:,opt_iter] = input_dic['slacks_2'][1][:,opt_iter].astype(float)/1000  
                    input_dic['slacks'][1][:,opt_iter] = input_dic['slacks'][1][:,opt_iter].astype(float)/1000  
                    # input_dic['Y_soln'][0][:,opt_iter] =   
                    input_dic['mu_cons2'][10:20,opt_iter] = input_dic['mu_cons2'][10:20,opt_iter].astype(float)/100  
                    input_dic['mu_cons1'][1:2,opt_iter] = input_dic['mu_cons1'][1:2,opt_iter].astype(float)/100  
                    input_dic['mu'][1:2,opt_iter] = input_dic['mu'][1:2,opt_iter].astype(float)/100 

                    input_dic['T_s'][0][:,opt_iter] = input_dic['T_s'][0][:,opt_iter].astype(float)/1000 
                    input_dic['Y_soln'][0][:,opt_iter] = input_dic['Y_soln'][0][:,opt_iter].astype(float)/1000   

                    print input_dic['T_n'][0][:,opt_iter]   
                    print input_dic['etaa'][0][:,opt_iter]    
                    print input_dic['uss'][0][:,opt_iter]    
                    print input_dic['slacks_2'][0][:,opt_iter]   
                    print input_dic['slacks'][0][:,opt_iter]    
                    # input_dic['Y_soln'][0][:,opt_iter] =   
                    print input_dic['mu_cons2'][0:10,opt_iter]   
                    print input_dic['mu_cons1'][0:1:,opt_iter]   
                    print input_dic['mu'][0:1,opt_iter]   

                    print input_dic['T_n'][1][:,opt_iter]   
                    print input_dic['etaa'][1][:,opt_iter]    
                    print input_dic['uss'][1][:,opt_iter]    
                    print input_dic['slacks_2'][1][:,opt_iter]   
                    print input_dic['slacks'][1][:,opt_iter]   
                    # input_dic['Y_soln'][1][:,opt_iter] =   
                    print input_dic['mu_cons2'][10:20,opt_iter]   
                    print input_dic['mu_cons1'][1:2,opt_iter]    
                    print input_dic['mu'][1:2,opt_iter]  

                    print input_dic['T_s'][0][:,opt_iter]   
                    print input_dic['Y_soln'][0][:,opt_iter]  

                    output_dic = sink5_dual.algorithm(node_id, opt_iter, input_dic)

                    print output_dic['T_n'][0][:,opt_iter]   
                    print output_dic['etaa'][0][:,opt_iter]    
                    print output_dic['uss'][0][:,opt_iter]    
                    print output_dic['slacks_2'][0][:,opt_iter]   
                    print output_dic['slacks'][0][:,opt_iter]    
                    # input_dic['Y_soln'][0][:,opt_iter] =   
                    print output_dic['mu_cons2'][0:10,opt_iter]   
                    print output_dic['mu_cons1'][0:1:,opt_iter]   
                    print output_dic['mu'][0:1,opt_iter]   

                    print output_dic['T_n'][1][:,opt_iter]   
                    print output_dic['etaa'][1][:,opt_iter]    
                    print output_dic['uss'][1][:,opt_iter]    
                    print output_dic['slacks_2'][1][:,opt_iter]   
                    print output_dic['slacks'][1][:,opt_iter]   
                    # input_dic['Y_soln'][1][:,opt_iter] =   
                    print output_dic['mu_cons2'][10:20,opt_iter]   
                    print output_dic['mu_cons1'][1:2,opt_iter]    
                    print output_dic['mu'][1:2,opt_iter]  

                    print output_dic['T_s'][0][:,opt_iter]   
                    print output_dic['Y_soln'][0][:,opt_iter]  

                    input_dic['T_s'][0][:,opt_iter] = ( output_dic['T_s'][0][:,opt_iter]*1000 ).astype(int) 
                    input_dic['Y_soln'][0][:,opt_iter] = ( output_dic['Y_soln'][0][:,opt_iter]*1000 ).astype(int)   

                    A['Node5'] = dummy
                    A['Node5'].message.decision_T[0:2] = input_dic['T_s'][node_id][:,opt_iter].astype(int).tolist()   
                    A['Node5'].message.Y_soln[0:2] = input_dic['Y_soln'][node_id][:,opt_iter].astype(int).tolist()  

                    # print output_dic['mu_cons2'][0:10,opt_iter]    
                    # print output_dic['mu_cons1'][0:1:,opt_iter]   
                    # print output_dic['mu'][0:1,opt_iter]   
                    # print output_dic['mu_cons2'][10:20,opt_iter]    
                    # print output_dic['mu_cons1'][1:2,opt_iter]   
                    # print output_dic['mu'][1:2,opt_iter] 
                           
                    parsed_data = A['Node5']
                    print parsed_data

                    ser.flush()
                    if (ser.isOpen()):
                        ser.write(motemessage2.build(parsed_data).encode('hex')+"\n")
                        print "Wrote to serial port ( dual update: sink 5 )"
                        
                    count = 0
                    print opt_iter
                    
                    # # print A['Node1'].message.decision_T[0:2]
                    # print input_dic['T_n'][0][:,opt_iter] 
                    # print input_dic['T_n'][1][:,opt_iter]
                    # print input_dic['T_s'][0][:,opt_iter]
                             
                #         max_constraints_2 = output_dic['max_constraints_2']     
                #         max_constraints_1 = output_dic['max_constraints_1']  
                #         max_sum_constraints = output_dic['max_sum_constraints']  
                #         obj_cvar_adl  = output_dic['obj_cvar_adl']  

                #         input_dic['max_constraints_2'] = output_dic['max_constraints_2']     
                #         input_dic['max_constraints_1'] = output_dic['max_constraints_1']  
                #         input_dic['max_sum_constraints'] = output_dic['max_sum_constraints']  
                #         input_dic['obj_cvar_adal'] = output_dic['obj_cvar_adal']             

ser.close()