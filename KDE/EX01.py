import numpy as np
import matplotlib.pyplot as plt

Data = np.load('data_ex02/meanshift1d.npy')
print(Data)
print(Data.shape)#(N,)


def mass_center(J,I):
    dist = (J[:,None]-I[None])**2
    Updated_J = np.array([np.mean(I[dist[j]<1]) for j in range(len(J))])
    return Updated_J

    # x_j in array J with shape(m,), x_i in array I with shape(n,)
    # Add new axis:(N,1)-(1,N) = (N,N)
    # A m*n matrix, M(j,i) represents the distance of node j and i.
    #For each j in J, we update j by Averaging the coordinate of nodes which is closer than 1.
    #I[dist[j]<1] gives back an array whose elements' corresponding place isless than 1:
    # The array contains x_i whose distance of X_j is less than 1.
    #Each iteration in j returns a number, so[] constitutes a list of x_j
    #Transfer to array

#This is only an one time updating J
def kernel(x, m, w):
    u = (x - m) / w
    supp = abs(u) < 1
    k = (1 - u ** 2) * 3 / (4 * w)
    return k * supp
def KDE(x, x_n, w):
    f = sum([kernel(x, m, w) for m in x_n]) / len(x_n)
    return f

x = np.arange(-5, 5, 0.0001)
#plt.plot(x, KDE(x, Data, 1))
#print('......', mass_center(Data, Data).shape)

t = 50 #The max step
trace = [Data, mass_center(Data, Data)]
for i in range(t):
    trace.append(mass_center(trace[-1],Data))
    #print('!!!!!', trace[-1].shape)
    #temp = mass_center(trace[-1], Data)
    #trace.append(temp)
    if np.all(trace[-1] == trace[-2]):
        break
#print(trace) why several times?
print(len(trace))#5
#print(trace[1].shape)#(20,)
trace = np.stack(trace) #Stack create a new axis 0 first, reshape (20,) to (1,20)
#And add them axis=0)
print('###',trace.shape)#(5,20)
plt.scatter(Data, np.zeros(len(Data))) #(Data,0)
print(trace.shape[1])#20
#print(trace[:, 1])
#print(range(len(trace)))#5
for i in range(trace.shape[1]):
    plt.scatter(trace[:, i], range(len(trace)), color = 'red')
    plt.plot(trace[:, i], range(len(trace)), color = 'darkblue', alpha = 0.2)
#len(trace) = trace.shape[0]
plt.ylabel('Nr of Update steps', fontsize = 20)
plt.xlabel('trace', fontsize = 20)

plt.show()