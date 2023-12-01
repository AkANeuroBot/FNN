
import matplotlib.pyplot as plt 
import numpy as np 
from net import Network, Punti
from matplotlib.animation import FuncAnimation
import os

os.chdir("home/THESIs/")

inf = float('inf')
epsilon = -np.pi
r_dom = 2*np.pi
punti = Punti(r_dom, epsilon) 
pp_gen = 1000 #numero di punti generati
x = punti.generatore(pp_gen).reshape(-1,1)
yy = np.exp(-(x**2)/2) * np.cos(4*x)*np.sin(4*x) + 0.3*np.sin(2*x) + 0.1*np.cos(4*x) #funzione da approssimare
num_input = 1
num_hidden = 128
num_output = 1
fft = np.fft.fft(yy)
EPOCHS = 20 #numero di epoche
lr = 1e-2
def cosine_squasher(k): #funzione di attivazione
        y = np.zeros(k.shape)
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                if k[i, j] < -np.pi/2: 
                    y[i, j] = 0
                elif k[i, j] > np.pi/2:
                    y[i, j] = 1
                else:
                    y[i, j] = (np.cos(k[i, j] + 3/2 * np.pi) + 1) / 2 

        return y
    
def cosine_squasher_deriv(k): #derivata della funzione di attivazione
        y = np.zeros(k.shape)
        
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                if k[i, j] < -np.pi/2 or k[i, j] > np.pi/2:
                    y[i, j] = 0
                else:
                    y[i, j] = -0.5 * np.sin(k[i, j] + 3/2 * np.pi)

        return y

class Neural_Network():
        def __init__(self,x,w1,w2,b1,b2, epoch, lr): 
            self.x = x
            self.w1 = w1
            self.w2 = w2
            self.b1 = b1
            self.b2 = b2
            self.EPOCHS = epoch
            self.lr = lr

        def forward(self,x,w1,w2,b1,b2):
            pred = np.zeros((self.EPOCHS, len(self.x)), dtype=np.float32)
            for epoch in range(self.EPOCHS):
                pred_array = np.array([], dtype=np.float32)
                for i in range(len(self.x)):
                    hidden_input = np.dot(x[i],w1)+b1 #input layer
                    hidden_output = cosine_squasher(hidden_input) 
                    prediction_F = (np.dot(hidden_output, w2)+b2)

                    loss_L2 = ((fft[i]-prediction_F) ** 2 )/2 #loss function
                    pred_array = np.append(pred_array, prediction_F[0][0] ).reshape(-1,1)

                    delta = prediction_F - fft[i]
                    deriv = cosine_squasher_deriv(hidden_input)
                    grad_hidden = np.dot(delta, w2.T)
                    grad_w2 = np.dot(hidden_output.T, delta)
                    grad_b2 = np.sum(delta, axis=0, keepdims=True)
                    grad_w1 = np.dot(x[i].T, grad_hidden*deriv)
                    grad_b1 = np.sum(grad_hidden*deriv, axis=0, keepdims=True)

                    #------------------------------------#
                    #--AGGIORNAMENTO DEI PESI------------#
                    #------------------------------------#
                    w1 = w1 - lr * grad_w1
                    b1 = b1 - lr * grad_b1
                    w2 = w2 - lr * grad_w2
                    b2 = b2 - lr * grad_b2

                pred[epoch] = pred_array.flatten()

            return pred
        

w1 = np.random.randn(num_input, num_hidden) 
b1 = np.zeros((1,num_hidden))
w2 = np.random.randn(num_hidden,num_output)
b2 = np.zeros((1,num_output))

net = Neural_Network(x, w1, w2, b1, b2, EPOCHS, lr)
fig = plt.figure()
y = net.forward(x,w1,w2,b1,b2)

fig = plt.figure()  
axis = plt.axes(xlim = (np.amin(x), np.amax(x)), ylim = (-1.5, 1.5))
axis.plot(x, yy, color = 'red', label = 'funzione da approssimare')
axis.set_title('Cosine Network')
axis.set_xlabel('x')
axis.set_ylabel('y')
axis.legend()

  
line, = axis.plot([], [], lw = 2, color = 'blue')  

def init():  
    line.set_data([], [])  
    return line,  
   
xdata, ydata = [], []  
   
# animation function  
def animate(i):  
    xdata = x
    ydata = y[i][:]
    line.set_data(xdata, ydata)  
      
    return line, 
ani = FuncAnimation(fig, animate,interval=800, blit=True, save_count=50, frames = EPOCHS*5) 
ani.save('cosine1.gif', writer='imagemagick', fps=60)
plt.show()



