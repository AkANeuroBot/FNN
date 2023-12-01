from net import *

dom = 2*np.pi
epsilon = 1E-6
dimension = 600
#punti = Punti(dom,epsilon)
num_input = 1
num_hidden = [3,5,10,20,50,70, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600]
num_output = 1
EPOCHS = 50 
lr = 1e-2 
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
effk = np.zeros((time,len(num_hidden)))
effc = np.zeros((time,len(num_hidden)))
lossk = np.zeros((time,len(num_hidden)))
lossc = np.zeros((time,len(num_hidden)))
epok = np.zeros((time,len(num_hidden)))
epoc = np.zeros((time,len(num_hidden)))

for t in range(time): 
    for i in num_hidden:
      
        net = Network(x,y,num_input,i,num_output,EPOCHS,lr)
        Cosine = net.Cosine_Network
        Logistic = net.logistic_Network
        print("------------------------------------")
        print(f'Neuroni strato nascosto: {i}')
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

        effk[t, num_hidden.index(i)] = R_cos[epoch_cos]
        effc[t, num_hidden.index(i) ] = R_log[epoch_log]
        lossk[t, num_hidden.index(i)] = loss_cos[epoch_cos]
        lossc[t, num_hidden.index(i)] = loss_log[epoch_log]
        epok[t, num_hidden.index(i)] = epoch_cos
        epoc[t, num_hidden.index(i)] = epoch_log


np.savetxt('effcos3.csv', effk, delimiter = ',')
np.savetxt('efflog3.csv', effc, delimiter = ',')
np.savetxt('losscos3.csv', lossk, delimiter = ',')
np.savetxt('losslog3.csv', lossc, delimiter = ',')
np.savetxt('epocos3.csv', epok, delimiter = ',')
np.savetxt('epolog3.csv', epoc, delimiter = ',')

