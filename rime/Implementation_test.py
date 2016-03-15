from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import numpy as np
import ADAL_test3

input_1 = np.zeros((2,4))
input_2 = np.zeros((2,1))
 
# Output = ADAL_test2.algorithm(input_1,input_2)
Output = ADAL_test3.algorithm(input_1,input_2)

max_iter_adal = 300

plt.figure(1)    
plt.plot(np.arange(max_iter_adal).reshape(1,max_iter_adal),Output['Obj'],'.b')

plt.figure(2)
plt.plot(np.arange(max_iter_adal).reshape(1,max_iter_adal),Output['Sum_Constraint'],'bo')
plt.plot(np.arange(max_iter_adal).reshape(1,max_iter_adal),Output['Constraint_1'],'rs')
plt.plot(np.arange(max_iter_adal).reshape(1,max_iter_adal),Output['Constraint_2'],'g^')
plt.yscale('log')
plt.show()

print Output['T_n'][0][:,10]  
print Output['T_n'][1][:,10] 
print Output['T_s'][0][:,10]  

print Output['etaa'][0][:,1]   
print Output['uss'][0][:,1]    
print Output['slacks_2'][0][:,1]     
print Output['slacks'][0][:,1]  

print Output['Y_soln'][0][:,1]    
print Output['mu_cons2'][0:10,1]     
print Output['mu_cons1'][0:1,1]     
print Output['mu'][0:1,1]  

print Output['etaa'][1][:,1]   
print Output['uss'][1][:,1]    
print Output['slacks_2'][1][:,1]     
print Output['slacks'][1][:,1]     
    
print Output['mu_cons2'][10:20,1]     
print Output['mu_cons1'][1:2,1]     
print Output['mu'][1:2,1]

 
 
