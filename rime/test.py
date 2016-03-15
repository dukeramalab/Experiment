from construct import *
import serial
import sys
import motemessage2
import time
import moteio

from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import numpy as np
import source1
import source1_dual
import math

def make2dList(rows, cols):
    a=[]
    for row in xrange(rows): a += [[0]*cols]
    return a

Ns = 10
J = 2
K = 1
lamba = 0.0
beta = 0.9
max_iter_adal = 1000
rho = 10
epsilon = 0.0
norm_option = 1

r_min_r = np.random.rand(J,1)
r_min_r = 0.2*np.ones(J).reshape(J,1)

R_r = []
R_r_indi = [] 
for v in range(Ns):
    R_r.append( np.random.rand(J,J+K) ) 
    R_r.append( 0.5*np.ones(J*(J+K)).reshape(J,J+K) )

    R_r[v][0,0] = 0
    R_r[v][1,1] = 0
    R_r[v][1,0] = 0
     
    R_r_indi.append( np.zeros((J,J+K)) )

R_mean = np.random.rand(J,J+K)
R_mean = 0.5*np.ones(J*(J+K)).reshape(J,J+K)

R_mean[0,0] = 0
R_mean[1,1] = 0
R_mean[1,0] = 0     

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

mycontrolstring = motemessage2.build(
    Container(
        message_type = 'CONTROL',
        part_number = 1,
        total_parts = 4,
        is_my_info = 1,
        message = Container(
            RSSI = 55,
            node_address = [10,105,118,140,42,12,5,0],
            eta = 0,
            uss = [0,0,0,0,0,0,0,0,0,0],
            slacks2 = [0,0,0,0,0,0,0,0,0,0],
            slacks = [0, 0],
            mu_cons2 = [0,0,0,0,0,0,0,0,0,0],
            mu_cons1 = 0,
            mu = 0,
            decision_T = [0,0,0,0],
            Y_soln = [0,0,0,0],
        ),
    ),
)
 
decoded_data = mycontrolstring 

A = {}
A['Node1'] = motemessage2.parse(mycontrolstring)

opt_iter = 0

node_id = 0

input_dic['T_n'][0][:,opt_iter] = np.asarray( A['Node1'].message.decision_T[0:2] )
input_dic['etaa'][0][:,opt_iter] = np.asarray( A['Node1'].message.eta )
input_dic['uss'][0][:,opt_iter] = np.asarray( A['Node1'].message.uss )
input_dic['slacks_2'][0][:,opt_iter] = np.asarray( A['Node1'].message.slacks2 )
input_dic['slacks'][0][:,opt_iter] = np.asarray( A['Node1'].message.slacks )
input_dic['Y_soln'][0][:,opt_iter] = np.asarray( A['Node1'].message.Y_soln[0:2] )
input_dic['mu_cons2'][0:10,opt_iter] = np.asarray( A['Node1'].message.mu_cons2 )
input_dic['mu_cons1'][0:1,opt_iter] = np.asarray( A['Node1'].message.mu_cons1 )
input_dic['mu'][0:1,opt_iter] = np.asarray( A['Node1'].message.mu ) 

output_dic = source1.algorithm(node_id, opt_iter, input_dic)

print np.transpose( output_dic['T_n'][0][:,opt_iter+1]*math.pow(10,0) )
print output_dic['etaa'][0][:,opt_iter+1]  
print output_dic['uss'][0][:,opt_iter+1] 
print output_dic['slacks_2'][0][:,opt_iter+1]  
print output_dic['slacks'][0][:,opt_iter+1]  
#print output_dic['Y_soln'][0][:,opt_iter+1]  
print output_dic['mu_cons2'][0:10,opt_iter+1]  
print output_dic['mu_cons1'][0:1,opt_iter+1]  
print output_dic['mu'][0:1,opt_iter+1]
print output_dic['mu'][0:1,opt_iter+1]*math.pow(10,9) != 0  
print 0*math.pow(10,9) != 0  

# a = ( np.random.rand(2,2)*(1000) ).astype(int)
# print a

# # print a.tolist()
# print math.pow(10,9)
# zz = 0.0022*1000
# print float(int(zz))/100
# print int(zz)/1000

 