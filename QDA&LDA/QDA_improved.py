'''
p(x|y=1) = N(x|-1,1**2)
p(x|y=2) = N(x|1,1**2)
assuming equal prior probability: p(y=1) = p(y=2)
'''
import numpy as np
import matplotlib.pyplot as plt
def normal_distribution(x, miu, sigma):
    y = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x-miu)/sigma)**2)
    return y

def posterior(lik_1, lik_2):
    return lik_1/(lik_1 + lik_2)

x = np.arange(-7,7,0.05)
#plt.plot(x, normal_distribution(x,-1,1))
#plt.show()
#posterior p(y=1|x)
post_1 = posterior(normal_distribution(x,-1,1),normal_distribution(x,1,1))
print(post_1)
post_2 = posterior(normal_distribution(x,1,1),normal_distribution(x,-1,1))
plt.plot(x,post_1,c = 'blue')
plt.plot(x,post_2,c = 'red')
plt.show()

post_1 = posterior(normal_distribution(x,-1,1.5),normal_distribution(x,1,1))
print(post_1)
post_2 = posterior(normal_distribution(x,1,1),normal_distribution(x,-1,1.5))
plt.plot(x,post_1,c = 'green')
plt.plot(x,post_2,c = 'yellow')
plt.show()




