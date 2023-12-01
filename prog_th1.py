import matplotlib.pyplot as plt
import random as r
import math as mt
import numpy as np
import timeit
from net import Network, Punti
inf = float('inf')
epsilon = -np.pi
pi = mt.pi
r_dom = 2*pi
punti = Punti(r_dom, epsilon) 
pp_gen = 1000 #numero di punti generati
x = punti.generatore(pp_gen).reshape(-1,1)
yy = np.exp(-(x**2)/2) * np.cos(4*x)*np.sin(4*x) + 0.3*np.sin(2*x) + 0.1*np.cos(4*x) #funzione da approssimare
num_input = 1
num_hidden = 128
num_output = 1
EPOCHS = 50 #numero di epoche
lr = 1e-2#learning rate
fft = np.fft.fft(yy)
net = Network(x, yy, num_input, num_hidden, num_output, EPOCHS, lr)
Cosine_Network = net.Cosine_Network
logistic_Network = net.logistic_Network
start = timeit.default_timer()
pred_cos, loss_cos, timer_cos, R_cos,epoch_cos = Cosine_Network()
stop = timeit.default_timer()
tempo = stop - start
print(f'Tempo di esecuzione del codice con cosine: {tempo}s')
start = timeit.default_timer()
pred_log, loss_log, timer_log, R_log, epoch_log = logistic_Network()
stop = timeit.default_timer()
tempo = stop - start
print(f'Tempo di esecuzione del codice con funzione logistica: {tempo}s')


fig, axs = plt.subplots(figsize=(12, 8))
axs.plot(range(epoch_cos+1),loss_cos, color='b', label='Cosine Network')
axs.plot(range(epoch_log+1),loss_log, color='r', label='Logistic Network')
axs.set_xlabel('Epochs')
axs.set_ylabel('Loss')
axs.set_title('Andamento della Loss nel tempo')
axs.legend()
plt.tight_layout
plt.savefig('Loss.png')
fig, axs = plt.subplots(figsize=(12, 8))
axs.plot(timer_cos,loss_cos, color='b', label='Cosine Network')
axs.plot(timer_log,loss_log, color='r', label='Logistic Network')
axs.set_xlabel('Time')
axs.set_ylabel('Loss')
axs.set_title('Andamento della Loss nel tempo')
axs.legend()
plt.tight_layout()
plt.savefig('Loss_time.png')
fig, axs = plt.subplots(figsize=(12, 8))
axs.plot(x, yy, label='Funzione iniziale', color='b')
axs.plot(x, pred_cos, label='Risultato Cosine_Net', color='g', linestyle='--')
axs.plot(x, pred_log, label='Risultato Logistic_Net', color='r', linestyle='--')
axs.legend()
axs.set_xlabel('x')
axs.set_ylabel('y')
axs.set_title('Confronto Funzione iniziale e Reti Neurali')
plt.tight_layout()
plt.savefig('Confronto.png')
fig, axs = plt.subplots(figsize=(12, 8))
axs.plot(x, yy, label='Funzione iniziale', color='b')
axs.plot(x,fft, label='FFT', color='r', linestyle='--')
axs.legend()
axs.set_xlabel('x')
axs.set_ylabel('y')
axs.set_title('Confronto Funzione iniziale e FFT')
plt.tight_layout()
plt.savefig('Adapt.png')
fig, axs = plt.subplots(figsize=(12, 8))
axs.plot(timer_cos, R_cos, color='b')
axs.plot(timer_log, R_log, color='r')
axs.set_xlabel('Tempo')
axs.set_ylabel('Efficienza')
axs.set_title('Andamento dell\'efficienza nel tempo')
axs.set_yscale('log')
axs.legend(['Cosine Network', 'Logistic Network'])
plt.tight_layout()
plt.savefig('Efficienza.png')
