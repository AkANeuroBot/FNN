import numpy as np
import matplotlib.pyplot as plt
from net import Network, Punti
dom = 2*np.pi
epsilon = 1E-6
dimension = [10,50,100,150,200,250,300,350,400,450,500,600,700,800,900,1000]
#Punti(dom,epsilon)
num_input = 1
num_hidden = 128
num_output = 1
EPOCHS = 50 
lr = 1e-2 
timerfunc_cos = np.array([],dtype = float)
timerfunc_log = np.array([],dtype = float)
efficienza_cos = np.array([],dtype = float)
efficienza_log = np.array([],dtype = float)
lossfunc_cos = np.array([],dtype = float)
lossfunc_log = np.array([],dtype = float)
epoccos = np.array([],dtype = float)
epoclog = np.array([],dtype = float)

LOSS_COS = np.zeros((len(dimension),EPOCHS),dtype = float)
TIME_COS = np.zeros((len(dimension),EPOCHS),dtype = float)
LOSS_LOG = np.zeros((len(dimension),EPOCHS),dtype = float)
TIME_LOG = np.zeros((len(dimension),EPOCHS),dtype = float)
time = 10

effk = np.zeros((time,len(dimension)))
effc = np.zeros((time,len(dimension)))
lossk = np.zeros((time,len(dimension)))
lossc = np.zeros((time,len(dimension)))
epok = np.zeros((time,len(dimension)))
epoc = np.zeros((time,len(dimension)))

for t in range(time):
    for dim in dimension:
        x = np.linspace(epsilon, dom-epsilon, dim)
        y = np.exp(-(x**2)/2) * np.cos(4*x)*np.sin(4*x) + 0.3*np.sin(2*x) + 0.1*np.cos(4*x)
        y = y.reshape(-1,1)
        net = Network(x,y,num_input,num_hidden,num_output,EPOCHS,lr)
        Cosine = net.Cosine_Network
        Logistic = net.logistic_Network
        print("------------------------------------")
        print(f'Dimensione dei punti: {dim}')
        print("-----------------COSINE SQUASHER-------------------")
        pred_cos, loss_cos, timer_cos, R_cos,epoch_cos = Cosine()
        print(f'Tempo di esecuzione: {timer_cos[epoch_cos]}s')
        print("-----------------LOGISTIC FUNC.-------------------")
        pred_log, loss_log, timer_log, R_log, epoch_log = Logistic()
        print(f'Tempo di esecuzione: {timer_log[epoch_log]}s')
        print("------------------------------------")

        timerfunc_cos = np.append(timerfunc_cos,timer_cos[epoch_cos])
        timerfunc_log = np.append(timerfunc_log,timer_log[epoch_log])
        efficienza_cos = np.append(efficienza_cos,R_cos[epoch_cos])
        efficienza_log = np.append(efficienza_log,R_log[epoch_log])
        lossfunc_cos = np.append(lossfunc_cos,loss_cos[epoch_cos])
        lossfunc_log = np.append(lossfunc_log,loss_log[epoch_log])
        epoccos = np.append(epoccos,epoch_cos)
        epoclog = np.append(epoclog,epoch_log)

        if loss_cos.shape != EPOCHS:
            LOSS_COS[dimension.index(dim)] = np.zeros(EPOCHS,dtype = float)
            TIME_COS[dimension.index(dim)] = np.zeros(EPOCHS,dtype = float)
        elif loss_log.shape != EPOCHS:
            LOSS_LOG[dimension.index(dim)] = np.zeros(EPOCHS,dtype = float)
            TIME_LOG[dimension.index(dim)] = np.zeros(EPOCHS,dtype = float)
        else:
            LOSS_COS[dimension.index(dim)] = loss_cos.reshape(-1)
            TIME_COS[dimension.index(dim)] = timer_cos.reshape(-1)
            LOSS_LOG[dimension.index(dim)] = loss_log.reshape(-1)
            TIME_LOG[dimension.index(dim)] = timer_log.reshape(-1)


        effk[t,dimension.index(dim)] = R_cos[epoch_cos]
        effc[t,dimension.index(dim)] = R_log[epoch_log]
        lossk[t,dimension.index(dim)] = loss_cos[epoch_cos]
        lossc[t,dimension.index(dim)] = loss_log[epoch_log]
        epok[t,dimension.index(dim)] = epoch_cos
        epoc[t,dimension.index(dim)] = epoch_log

np.savetxt('effcos2.csv',effk, delimiter=',', fmt='%f')
np.savetxt('efflog2.csv',effc, delimiter=',', fmt='%f')
np.savetxt('losscos2.csv',lossk, delimiter=',', fmt='%f')
np.savetxt('losslog2.csv',lossc, delimiter=',', fmt='%f')
np.savetxt('epocos2.csv',epok, delimiter=',', fmt='%f')
np.savetxt('epolog2.csv',epoc, delimiter=',', fmt='%f')

