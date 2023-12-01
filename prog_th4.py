from net import *

dom = 2*np.pi
epsilon = 1E-6
dimension = 600
#punti = Punti(dom,epsilon)
num_input = 1
num_hidden = 128
num_output = 1
EPOCHS = 50 
lr = [1e-1,1e-2,5e-2,1e-3,5e-3,1e-4,5e-4,1e-5] 
lr.sort()

x = np.linspace(epsilon, dom-epsilon, dimension)
y = np.exp(-(x**2)/2) * np.cos(4*x)*np.sin(4*x) + 0.3*np.sin(2*x) + 0.1*np.cos(4*x)
y = y.reshape(-1,1)
efficienza_cos = np.array([],dtype = float)
efficienza_log = np.array([],dtype = float)
lossfunc_cos = np.array([],dtype = float)
lossfunc_log = np.array([],dtype = float)
epoccos = np.array([],dtype = float)
epoclog = np.array([],dtype = float)
time = 10

effk = np.zeros((time,len(lr)))
effc = np.zeros((time,len(lr)))
lossk = np.zeros((time,len(lr)))
lossc = np.zeros((time,len(lr)))
epok = np.zeros((time,len(lr)))
epoc = np.zeros((time,len(lr)))

for t in range(time):

    for i in lr:
        net = Network(x,y,num_input,num_hidden,num_output,EPOCHS,i)
        Cosine = net.Cosine_Network
        Logistic = net.logistic_Network
        print("------------------------------------")
        print(f'valore learning rate: {i}')
        print("-----------------COSINE SQUASHER-------------------")
        pred_cos, loss_cos, timer_cos, R_cos,epoch_cos = Cosine()
        print("-----------------LOGISTIC FUNC.-------------------")
        pred_log, loss_log, timer_log, R_log, epoch_log = Logistic()
        print("------------------------------------")
        efficienza_cos = np.append(efficienza_cos,R_cos[epoch_cos])
        efficienza_log = np.append(efficienza_log,R_log[epoch_log])
        lossfunc_cos = np.append(lossfunc_cos,loss_cos[epoch_cos])
        lossfunc_log = np.append(lossfunc_log,loss_log[epoch_log])
        epoccos = np.append(epoccos,epoch_cos)
        epoclog = np.append(epoclog,epoch_log)

        effk[t,lr.index(i)] = R_cos[epoch_cos] 
        effc[t,lr.index(i)] = R_log[epoch_log]
        lossk[t,lr.index(i)] = loss_cos[epoch_cos]
        lossc[t,lr.index(i)] = loss_log[epoch_log]
        epok[t,lr.index(i)] = epoch_cos
        epoc[t,lr.index(i)] = epoch_log

np.savetxt('effcos4.csv', effk, delimiter = ',')
np.savetxt('efflog4.csv', effc, delimiter = ',')
np.savetxt('losscos4.csv', lossk, delimiter = ',')
np.savetxt('losslog4.csv', lossc, delimiter = ',')
np.savetxt('epocos4.csv', epok, delimiter = ',')
np.savetxt('epolog4.csv', epoc, delimiter = ',')

