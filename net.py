import numpy as np
import timeit
import matplotlib.pyplot as plt
from matplotlib import animation

class Punti():
    def __init__(self, dom, epsilon):
        self.dom = dom
        self.epsilon = epsilon
    
    #genera un numero di punti equidistanti nell'intervallo [epsilon, dom-epsilon]
    def generatore(self, num):
        step = (self.dom - self.epsilon) / num
        result = [self.epsilon + i * step for i in range(num)]
        return np.array([result])
    
class Network:
    def __init__(self, x, yy, num_input, num_hidden, num_output, epoc, lr):
        self.x = x
        self.yy = yy
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.EPOCHS = epoc
        self.lr = lr
        self.weight = self.pesi
        self.cos = self.cosine_squasher
        self.cos_deriv = self.cosine_squasher_deriv
        self.log = self.logistic_function
        self.log_deriv = self.logistic_function_deriv

    def logistic_function_deriv(self,k):
        y = np.zeros(k.shape)
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                y[i, j] = np.exp(-k[i, j]) / (1 + np.exp(-k[i, j]))**2
        return y
    
    def logistic_function(self,k):
        y = np.zeros(k.shape)
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                y[i, j] = 1 / (1 
                               + np.exp(-k[i, j]))
        return y

    def cosine_squasher(self, k): #funzione di attivazione
        y = np.zeros(k.shape)
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                if k[i, j] < -np.pi/2: 
                    y[i, j] = 0
                elif k[i, j] > np.pi/2:
                    y[i, j] = 1
                else:
                    y[i, j] = (np.cos(k[i, j] 
                                      + 3/2 * np.pi) 
                                      + 1) / 2 
        return y
    
    def cosine_squasher_deriv(self,k): #derivata della funzione di attivazione
        y = np.zeros(k.shape)
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                if k[i, j] < -np.pi/2 or k[i, j] > np.pi/2:
                    y[i, j] = 0
                else:
                    y[i, j] = -0.5 * np.sin(k[i, j] 
                                            + 3/2 * np.pi)
        return y

    def pesi(self, num_i, num_h): #inizializza casualmente i pesi per la rete neurale
        w = np.random.randn(num_i, num_h)
        return w
    

    def NN_cos(self, w1, b1, w2, b2, i): #Rete Neurale con cosine squasher
        hidden_input = np.dot(self.x[i],w1)+b1 #input layer
        hidden_output = self.cos(hidden_input) 
        prediction_F = (np.dot(hidden_output, w2)+b2)


        delta = prediction_F - self.yy[i]
        deriv = self.cos_deriv(hidden_input)
        grad_hidden = np.dot(delta, w2.T)
        grad_w2 = np.dot(hidden_output.T, delta)
        grad_b2 = np.sum(delta, axis=0, keepdims=True)
        grad_w1 = np.dot(self.x[i].T, grad_hidden*deriv)
        grad_b1 = np.sum(grad_hidden*deriv, axis=0, keepdims=True)

        w1 = w1 - self.lr * grad_w1
        b1 = b1 - self.lr * grad_b1
        w2 = w2 - self.lr * grad_w2
        b2 = b2 - self.lr * grad_b2

        return prediction_F, w1, b1, w2, b2

    def NN_log(self, w1, b1, w2, b2, i): #Rete Neurale con funzione logistica

        hidden_input = np.dot(self.x[i],w1)+b1
        hidden_output = self.log(hidden_input) 
        prediction_F = (np.dot(hidden_output, w2)+b2)

        delta = prediction_F - self.yy[i]
        deriv = self.log_deriv(hidden_input)
        grad_hidden = np.dot(delta, w2.T)
        grad_w2 = np.dot(hidden_output.T, delta)
        grad_b2 = np.sum(delta, axis=0, keepdims=True)
        grad_w1 = np.dot(self.x[i].T, grad_hidden*deriv)
        grad_b1 = np.sum(grad_hidden*deriv, axis=0, keepdims=True)

        w1 = w1 - self.lr * grad_w1
        b1 = b1 - self.lr * grad_b1
        w2 = w2 - self.lr * grad_w2
        b2 = b2 - self.lr * grad_b2

        return prediction_F, w1, b1, w2, b2


    def Cosine_Network(self): #Algoritmo di apprendimento
        losses = np.array([], dtype=np.float32)
        start = timeit.default_timer()
        timer = np.array([], dtype=np.float32)
        R = np.array([], dtype=np.float32)
        w1 = self.weight(self.num_input, self.num_hidden) 
        b1 = np.zeros((1,self.num_hidden))
        w2 = self.weight(self.num_hidden,self.num_output)
        b2 = np.zeros((1,self.num_output))
       
        for epoch in range(self.EPOCHS):
            pred_array = np.array([], dtype=np.float32)
            for i in range(len(self.x)):
                pred, w1, b1, w2, b2 = self.NN_cos(w1, b1, w2, b2, i)
                pred_array = np.append(pred_array, pred[0][0] ).reshape(-1,1)


            sst = np.sum(self.yy **2)
            ssr = np.sum((self.yy - pred_array)**2)
            r2 = 1 - ssr/sst
            if r2 < 0:
                r2 = 0
    
            R = np.append(R, round(r2,6)*100)
            loss = 1/2*np.sum((pred_array - self.yy)**2)
            losses = np.append(losses, loss)
            stop_epoch = timeit.default_timer()
            timer = np.append(timer, stop_epoch - start)

            if epoch % 10 == 0 :
                    sst = np.sum(self.yy **2)
                    ssr = np.sum((self.yy - pred_array)**2)
                    r2 = 1 - ssr/sst
                    if r2 < 0:
                        r2 = 0
                    stop = timeit.default_timer()
                    print(f'Epoch: {epoch} | Loss: {loss} | Efficienza: {round(r2,6)*100}%')
            if r2 > 0.999:
                stop = timeit.default_timer()
                print(f'Epoch: {epoch}| Loss: {loss}  | Efficienza: {round(r2,6)*100}%')
                break
        return pred_array, losses, timer, R, epoch

    def logistic_Network(self): #Algoritmo di apprendimento
        losses = np.array([], dtype=np.float32)
        start = timeit.default_timer()
        timer = np.array([], dtype=np.float32)
        R = np.array([], dtype=np.float32)
        w1 = self.weight(self.num_input, self.num_hidden) 
        b1 = np.zeros((1,self.num_hidden))
        w2 = self.weight(self.num_hidden,self.num_output)
        b2 = np.zeros((1,self.num_output))
        
        for epoch in range(self.EPOCHS):
            pred_array = np.array([], dtype=np.float32)
            for i in range(len(self.x)):
                pred, w1, b1, w2, b2 = self.NN_log(w1, b1, w2, b2, i)
                pred_array = np.append(pred_array, pred[0][0] ).reshape(-1,1)
            
            sst = np.sum(self.yy **2)
            ssr = np.sum((self.yy - pred_array)**2)
            r2 = 1 - ssr/sst

            if r2 < 0:
                r2 = 0
            R = np.append(R, round(r2,6)*100)
            loss = 1/2*np.sum((pred_array - self.yy)**2)
            losses = np.append(losses, loss)
            stop_epoch = timeit.default_timer()
            timer = np.append(timer, stop_epoch - start)

            if epoch % 10 == 0 :
                    ssr = np.sum((self.yy - pred_array)**2)
                    r2 = 1 - ssr

                    if r2 < 0:
                        r2 = 0
                    print(f'Epoch: {epoch} | Loss: {loss} | Efficienza: {round(r2,6)*100}%')
            if r2 > 0.999:
                print(f'Epoch: {epoch} | Loss: {loss} | Efficienza: {round(r2,6)*100}%')
                break
        return pred_array, losses, timer, R, epoch
    
