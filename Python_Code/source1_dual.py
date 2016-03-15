from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import numpy as np

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
    #R_r.append( np.random.rand(J,J+K) ) 
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

w = np.ones((K,1))



def algorithm(node_id, opt_iter, input_dic):

    opt_iter = opt_iter - 1
    #opt_iter = opt_iter + 1

    i = node_id

    T_n = input_dic['T_n']   
    etaa = input_dic['etaa']  
    uss = input_dic['uss']  
    slacks_2 = input_dic['slacks_2']     
    slacks = input_dic['slacks']    
    T_s = input_dic['T_s']   
    Y_soln = input_dic['Y_soln']   
    mu_cons2 = input_dic['mu_cons2']     
    mu_cons1 = input_dic['mu_cons1']  
    mu = input_dic['mu']  
    constraints_2 = input_dic['constraints_2']     
    constraints_1 = input_dic['constraints_1']  
    sum_constraints = input_dic['sum_constraints'] 
    max_constraints_2  = input_dic['max_constraints_2']  
    max_constraints_1 = input_dic['max_constraints_1'] 
    max_sum_constraints = input_dic['max_sum_constraints'] 
    obj_cvar_adal = input_dic['obj_cvar_adal']  
                    
    # Dual updates
    # Begin sum flow
    RES = -np.ones(J).reshape(J,1)
    for j in range(J):
        RES[j,0] = RES[j,0] + sum( T_n[j][:,opt_iter+1] ) + slacks[j][1,opt_iter+1]

    for j in range(K):
        RES = RES + T_s[j][:,opt_iter+1].reshape(J,1)

    sum_constraints[:,opt_iter] = RES.reshape(J)
    max_sum_constraints[:,opt_iter] = max(abs( RES )).reshape(1)

    mu[:,opt_iter+1] = mu[:,opt_iter] + tau*rho*sum_constraints[:,opt_iter]
    # End sum flow

    # Begin network risk
    TT_temp = np.zeros((J,J+K)) 
    for j in range(K):
        TT_temp[:,J+j] = T_s[j][:,opt_iter+1]

    QQ_3_r = np.zeros((J,Ns))
    for scen in range(Ns):
        QQ_3_r[:,scen] = np.zeros((J,1)).reshape(J)
        for j in range(J):
            QQ_3_r[:,scen] = QQ_3_r[:,scen] + np.dot( RR_r[j][scen], T_n[j][:,opt_iter+1].reshape(J,1) ).reshape(J)
            QQ_3_r[j,scen] = QQ_3_r[j,scen] + r_min_r[j,0]  

    RTs_3 = np.zeros((J,Ns))
    for scen in range(Ns):
        RTs_3[:,scen] = QQ_3_r[:,scen]
        for jj in range(J):
            constraints_2[ (jj)*Ns+scen,opt_iter] = ( -etaa[jj][0,opt_iter+1]-uss[jj][scen,opt_iter+1]
                +RTs_3[jj,scen]-np.sum( R_r[scen][jj,J:J+K]*TT_temp[jj,J:J+K] )+slacks_2[jj][scen,opt_iter+1] )

    max_constraints_2[:,opt_iter] = max(abs( constraints_2[:,opt_iter] )).reshape(1)

    mu_cons2[:,opt_iter+1] = mu_cons2[:,opt_iter] + tau*rho*constraints_2[:,opt_iter]      
    # End network risk

    # Begin network mean
    QQ_3_b = np.zeros((J,1))
    for j in range(J):
        QQ_3_b[:,0] = QQ_3_b[:,0] + np.dot( RR_mean[j][0], T_n[j][:,opt_iter+1].reshape(J,1) ).reshape(J) 
        QQ_3_b[j,0] = QQ_3_b[j,0] + r_min_r[j,0]  
        QQ_3_b[j,0] = QQ_3_b[j,0] + slacks[j][0,opt_iter+1]

    for j in range(K):
        QQ_3_b = QQ_3_b - (R_mean[:,J+j]*T_s[j][:,opt_iter+1]).reshape(J,1)

    constraints_1[:,opt_iter] = QQ_3_b.reshape(J)
    max_constraints_1[:,opt_iter] = max(abs( constraints_1[:,opt_iter] )).reshape(1)

    mu_cons1[:,opt_iter+1] = mu_cons1[:,opt_iter] + tau*rho*constraints_1[:,opt_iter] 
    # End network mean

    # Collect data
    TT = np.zeros((J,J+K))
    for a_i in range(J):
        TT[a_i,0:J] = T_n[a_i][:,opt_iter+1]

    cost_first_y = np.zeros((J,1))
    for a_i in range(K):
        TT[:,J+a_i] = T_s[a_i][:,opt_iter+1]
        cost_first_y[a_i,0] = w[a_i,0]*sum( Y_soln[a_i][:,opt_iter+1] )

    ETA = np.zeros((J,1))
    US = np.zeros((J,Ns))
    for scen in range(Ns):
        for jj in range(J):
            ETA[jj,0] = etaa[jj][:,opt_iter+1]
            US[jj,scen] = uss[jj][scen,opt_iter+1]

    obj_cvar_adal[0,opt_iter] = lamba*np.sum(cost_first_y) + (1-lamba)*np.sum( ETA + (1/Ns)*( 1/(1-beta) )*np.sum(US,1).reshape(J,1) )

    output_dic = {}
    output_dic['T_n'] = T_n  
    output_dic['T_s'] = T_s
    output_dic['etaa'] = etaa  
    output_dic['uss'] = uss
    output_dic['slacks_2'] = slacks_2    
    output_dic['slacks'] = slacks     
    output_dic['Y_soln'] = Y_soln  
    output_dic['mu_cons2'] = mu_cons2    
    output_dic['mu_cons1'] = mu_cons1 
    output_dic['mu'] = mu 
    output_dic['constraints_2'] = constraints_2    
    output_dic['constraints_1'] = constraints_1 
    output_dic['sum_constraints'] = sum_constraints 
    output_dic['max_constraints_2'] = max_constraints_2    
    output_dic['max_constraints_1'] = max_constraints_1
    output_dic['max_sum_constraints'] = max_sum_constraints
    output_dic['obj_cvar_adal'] = obj_cvar_adal   
        
    return output_dic