from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import numpy as np

def make2dList(rows, cols):
    a=[]
    for row in xrange(rows): a += [[0]*cols]
    return a

def algorithm(T1, T2):
    
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
        # T_n.append( np.random.rand(J,max_iter_adal+1) )
        T_n.append( np.zeros((J,max_iter_adal+1)) )
        etaa.append( np.zeros((1,max_iter_adal+1)) )
        uss.append( np.zeros((Ns,max_iter_adal+1)) )
        slacks_2.append( np.zeros((Ns,max_iter_adal+1)) )
        slacks.append( np.zeros((2,max_iter_adal+1)) )

    T_s = [] 
    Y_soln = []
    for j in range(K):
        # T_s.append( np.random.rand(J,max_iter_adal+1) )
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

    for opt_iter in range(max_iter_adal):
                           
        w = np.ones((K,1))
    
        # Primal updates (source nodes)
        G = np.concatenate( ( np.eye(J+J+1+1+Ns+Ns+2), -np.eye(J+J+1+1+Ns+Ns+2 ) ), axis=0 )
        h = np.concatenate( ( np.ones(J+J+1).reshape(J+J+1,1), 10*np.ones(1+Ns+Ns+2).reshape(1+Ns+Ns+2,1) ), axis=0 )
        h = np.concatenate( ( h, 0*np.ones(J+J+1+1+Ns+Ns+2).reshape(J+J+1+1+Ns+Ns+2,1) ), axis=0 )
        h[J+J+1+1+Ns+Ns+2+J+J+1,0] = 1000
        A = np.concatenate( (np.zeros((J+1,J)), np.diagflat(np.ones((J+1,1))), np.zeros((J+1,1)), np.zeros((J+1,Ns)), np.zeros((J+1,Ns)), np.zeros((J+1,2))), axis = 1 ) 
        b = np.ones((J+1,1))
    
        G = matrix(G)
        h = matrix(h)
        A = matrix(A)
        b = matrix(b)
    
        for i in range(J):
            QQ = r_min_r
            for j in range(J):
                if j != i:
                    QQ = QQ + np.dot( RR_mean[j][0], T_n[j][:,opt_iter].reshape(J,1) )
                    QQ[j,0] = QQ[j,0] + slacks[j][0,opt_iter]
            for j in range(K):
                QQ = QQ - (R_mean[:,J+j].reshape(J,1))*(T_s[j][:,opt_iter].reshape(J,1))
            
            SS = -1
            for j in range(K):
                SS = SS + T_s[j][i,opt_iter]
        
            QQ_2_r = np.zeros((J,Ns)) 
            for scen in range(Ns):
                QQ_2_r[:,scen] = r_min_r.reshape(J)
                for j in range(J):
                    if j != i:
                        QQ_2_r[:,scen] = QQ_2_r[:,scen] + np.dot( RR_r[j][scen], T_n[j][:,opt_iter].reshape(J,1) ).reshape(J)
                        QQ_2_r[j,scen] = QQ_2_r[j,scen] - etaa[j][0,opt_iter]- uss[j][scen,opt_iter] + slacks_2[j][scen,opt_iter] 
                        
            for scen in range(Ns):
                for j in range(K):
                    QQ_2_r[:,scen] = QQ_2_r[:,scen] - R_r[scen][:,J+j]*T_s[j][:,opt_iter] 
                
            QQ_2_r_stack = np.diagflat(QQ)
            for scen in range(Ns):
                QQ_2_r_stack = np.concatenate( ( QQ_2_r_stack, np.diagflat(QQ_2_r[:,scen].reshape(J,1)) ), axis=0 ) 
            
            RR_r_stack = RR_mean[i][0]
            for scen in range(Ns):
                RR_r_stack = np.concatenate( ( RR_r_stack, RR_r[i][scen] ), axis = 0 )
        
            ee = np.zeros((J*(Ns+1),1))
            for scen in range(Ns):
                ee[ (scen+1)*J+i,0 ] = 1
        
            ee_0 = np.zeros((J*(Ns+1),1))
            ee_0[i,0] = 1
        
            ee_0_b = np.zeros((J,1))
            ee_0_b[i,0] = 1
        
            EE = np.zeros((J*(Ns+1),1))
            DIAG_EE = np.zeros((J*(Ns+1),Ns))
            for scen in range(Ns):
                EE[ (scen+1)*J+i,0  ] = 1;
                DIAG_EE[ (scen+1)*J:(scen+2)*J,scen ] = EE[ (scen+1)*J:(scen+2)*J,0 ];
        
            mu_cons2_temp = np.zeros((J,Ns))
            for scen in range(Ns):
                for j in range(J):
                    mu_cons2_temp[j,scen] = mu_cons2[ (j)*Ns+scen, opt_iter ];
        
            F_4 = np.dot( mu_cons2_temp[:,0].reshape(J,1).T, ee_0_b ) - (1-lamba)/(Ns*(1-beta))
            F_5 = np.dot( mu_cons2_temp[:,0].reshape(J,1).T, ee_0_b )
            f_1 = np.zeros((Ns,J))
            f_3 = np.zeros((Ns,1))
            for scen in range(Ns):
                f_1[scen,:] = np.dot(mu_cons2_temp[:,scen].reshape(J,1).T, RR_r[i][scen] ).reshape(J)
                f_3[scen,0] = np.dot( mu_cons2_temp[:,scen].reshape(J,1).T, ee_0_b )
        
            for scen in range(Ns-1):
                F_4 = np.concatenate( (F_4, np.dot( mu_cons2_temp[:,scen+1].reshape(J,1).T, ee_0_b ) - (1-lamba)/(Ns*(1-beta)) ), axis = 1 )
                F_5 = np.concatenate( (F_5, np.dot( mu_cons2_temp[:,scen+1].reshape(J,1).T, ee_0_b ) ), axis = 1 )  
        
            p = np.concatenate( ( np.sum(f_1,0)+np.dot(mu_cons1[:,opt_iter].T,RR_mean[i][0])+mu[i,opt_iter]*np.ones((1,J)), 
                np.zeros((1,J+1)),
                (-np.sum(f_3,0)+(1-lamba)).reshape(1,1),
                -F_4,
                F_5,
                np.dot(mu_cons1[:,opt_iter].reshape(J,1).T,ee_0_b).reshape(1,1),
                mu[i,opt_iter].reshape(1,1) ), axis=1 ).T  
            
            Q_h_1 = np.concatenate( ( RR_r_stack, QQ_2_r_stack, np.zeros((J*(Ns+1),1)), -ee, -DIAG_EE, DIAG_EE, ee_0, np.zeros((J*(Ns+1),1)) ), axis=1 )
            Q_h_2 = np.concatenate( ( np.ones((1,J)), np.zeros((1,J)), SS.reshape(1,1), np.zeros((1,1+Ns+Ns+1)), np.ones((1,1)) ), axis=1 )      
            Q = rho*np.dot( np.concatenate( (Q_h_1,Q_h_2), axis=0 ).T, np.concatenate( (Q_h_1,Q_h_2), axis=0 ) )
        
            Q = matrix(Q)
            p = matrix(p)
        
            solvers.options['show_progress'] = False
            sol=solvers.qp(Q, p, G, h, A, b)
            Z = np.array(sol['x'])
        
            T_n[i][:,opt_iter+1] = T_n[i][:,opt_iter] + tau*( Z[0:J,0] - T_n[i][:,opt_iter] )  
            etaa[i][:,opt_iter+1] = etaa[i][:,opt_iter] + tau*( Z[J+J+1,0] - etaa[i][:,opt_iter] )
            uss[i][:,opt_iter+1] = uss[i][:,opt_iter] + tau*( Z[J+J+1+1:J+J+1+1+Ns,0] - uss[i][:,opt_iter] )
            slacks_2[i][:,opt_iter+1] = slacks_2[i][:,opt_iter] + tau*( Z[J+J+1+1+Ns:J+J+1+1+Ns+Ns,0] - slacks_2[i][:,opt_iter] )
            slacks[i][:,opt_iter+1] = slacks[i][:,opt_iter] + tau*( Z[J+J+1+1+Ns+Ns:,0] - slacks[i][:,opt_iter] )                              
    
        EYE_BLK_1 = np.concatenate( ( np.eye(J,J), -np.eye(J,J) ), axis=0 )
        EYE_BLK_2 = np.concatenate( ( -np.eye(J,J), -np.eye(J,J) ), axis=0 )
    
        G_1 = np.concatenate( ( np.eye(J+J+1+J), -np.eye(J+J+1+J) ), axis=0 )
        G_2 = np.concatenate( ( EYE_BLK_1, np.zeros((2*J,J+1)), EYE_BLK_2 ), axis = 1) 
        G = np.concatenate( (G_1, G_2), axis=0 )
        h_1 = np.concatenate( ( np.ones(J+J+1+J).reshape(J+J+1+J,1), 0*np.ones(J+J+1+J).reshape(J+J+1+J,1) ), axis=0 )
        h_2 = np.zeros((2*J,1))
        h = np.concatenate( (h_1, h_2), axis=0 )
        A = np.concatenate( ( np.zeros((J+1,J)), np.diagflat(np.ones((J+1,1))), np.zeros((J+1,J)) ), axis = 1 ) 
        b = np.ones((J+1,1))
    
        G = matrix(G)
        h = matrix(h)
        A = matrix(A)
        b = matrix(b)
    
        # Primal updates (sink nodes)
        for i in range(K):
            QQ_3_b = np.zeros((J,1))
            for j in range(J):
                QQ_3_b = QQ_3_b + np.dot( RR_mean[j][0], T_n[j][:,opt_iter].reshape(J,1) )
                QQ_3_b[j,0] = QQ_3_b[j,0] + r_min_r[j,0]  
                QQ_3_b[j,0] = QQ_3_b[j,0] + slacks[j][0,opt_iter]
            
            for j in range(K):
                if j != i:
                    QQ_3_b = QQ_3_b - ( R_mean[:,J+j].reshape(J,1) )*( T_s[j][:,opt_iter].reshape(J,1) )
        
            QQ_3_r = np.zeros((J,Ns))
            for scen in range(Ns):
                for j in range(J):
                    QQ_3_r[:,scen] = QQ_3_r[:,scen] + np.dot( RR_r[j][scen], T_n[j][:,opt_iter].reshape(J,1) ).reshape(J)
                    QQ_3_r[j,scen] = QQ_3_r[j,scen] + r_min_r[j,0] 
                    QQ_3_r[j,scen] = QQ_3_r[j,scen] -etaa[j][0,opt_iter]- uss[j][scen,opt_iter] + slacks_2[j][scen,opt_iter]
        
            for scen in range(Ns):
                for j in range(K):
                    if j != i:
                        QQ_3_r[:,scen] = QQ_3_r[:,scen] - R_r[scen][:,J+j]*T_s[j][:,opt_iter]
                    
            QQ_3_r_stack = np.diagflat( QQ_3_b )
            for scen in range(Ns):
                QQ_3_r_stack = np.concatenate( (QQ_3_r_stack, np.diagflat(QQ_3_r[:,scen].reshape(J,1))), axis=0 )  
            
            RR_r_stack = np.diagflat( -R_mean[:,J+i].reshape(J,1) )
            for scen in range(Ns):
                RR_r_stack = np.concatenate( ( RR_r_stack, np.diagflat( -R_r[scen][:,J+i].reshape(J,1) ) ), axis=0 )
        
            SS_b = -np.ones(J).reshape(J,1)
            for j in range(K):
                if j != i:
                    SS_b = SS_b + T_s[j][:,opt_iter].reshape(J,1)
            for j in range(J):
                SS_b[j,0] = np.sum( T_n[j][:,opt_iter] ) + slacks[j][1,opt_iter]
        
            mu_cons2_temp = np.zeros((J,Ns))
            for scen in range(Ns):
                for j in range(J):
                    mu_cons2_temp[j,scen] = mu_cons2[ (j)*Ns+scen,opt_iter ]
        
            f_1 = np.zeros((Ns,J))
            for scen in range(Ns):
                f_1[scen,:] = mu_cons2_temp[:,scen]*R_r[scen][:,J+i]  
            
            p = np.concatenate( ( -np.sum(f_1,0)-(mu_cons1[:,opt_iter]*R_mean[:,J+i]).reshape(J,1).T+mu[:,opt_iter].reshape(J,1).T, 
            np.zeros((1,J+1)),
            lamba*w[i,0]*np.ones((1,J)) ), axis=1 ).T   
        
            Q_h_1 = np.concatenate( ( RR_r_stack, QQ_3_r_stack, np.zeros((J*(Ns+1),1+J)) ), axis=1 )
            Q_h_2 = np.concatenate( ( np.eye(J), np.zeros((J,J)), SS_b, np.zeros((J,J)) ), axis=1 )
            Q = rho*np.dot( np.concatenate( (Q_h_1,Q_h_2), axis=0 ).T, np.concatenate( (Q_h_1,Q_h_2), axis=0 ) )
     
            Q = matrix(Q)
            p = matrix(p)
        
            sol=solvers.qp(Q, p, G, h, A, b)
        
            Z = np.array(sol['x'])
        
            T_s[i][:,opt_iter+1] = T_s[i][:,opt_iter] + tau*( Z[0:J,0] - T_s[i][:,opt_iter] ) 
            Y_soln[i][:,opt_iter+1] = Z[J+J+1:,0]
         
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

        decision_T = np.zeros((1,4)) 
        decision_T = np.ones(4).reshape(1,4)

        Output = {}
        Output['Obj'] = obj_cvar_adal
        Output['Sum_Constraint'] = max_sum_constraints
        Output['Constraint_1'] = max_constraints_1   
        Output['Constraint_2'] = max_constraints_2 

    return Output


 


