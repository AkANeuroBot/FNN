import numpy as np 
import matplotlib.pyplot as plt

dimension = [10,50,100,150,200,250,300,350,400,450,500,600,700,800,900,1000]
num_hidden = [3,5,10,20,50,70, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600]
lr = [1e-1,1e-2,5e-2,1e-3,5e-3,1e-4,5e-4,1e-5] 

data = [dimension, num_hidden, lr]

nameeffc = ['effcos2.csv', 'effcos3.csv', 'effcos4.csv']
namelossc = ['losscos2.csv', 'losscos3.csv', 'losscos4.csv']
nameepoc = ['epocos2.csv', 'epocos3.csv', 'epocos4.csv']
nameeffl = ['efflog2.csv', 'efflog3.csv', 'efflog4.csv']
namelossl = ['losslog2.csv', 'losslog3.csv', 'losslog4.csv']
nameepol = ['epolog2.csv', 'epolog3.csv', 'epolog4.csv']

name_file_eff = ['eff-dim_rat.png', 'eff-nh_rat.png', 'eff-lr_rat.png']
name_file_loss = ['loss-dim_rat.png', 'loss-nh_rat.png', 'loss-lr_rat.png']
name_file_epoch = ['epoche-dim_rat.png', 'epoche-nh_rat.png', 'epoche-lr_rat.png']
for nameff_co,nameloss_co,nameepoc_co, nameff_lo,nameloss_lo,nameepoc_lo,data, name_eff, name_loss, name_epoc in zip(nameeffc,namelossc,nameepoc,nameeffl,namelossl,nameepol,data, name_file_eff, name_file_loss, name_file_epoch):
    
    effk = np.loadtxt(nameff_co, delimiter = ','
                    , dtype = float)
    effc = np.loadtxt(nameff_lo, delimiter = ','
                    , dtype = float)
    lossk = np.loadtxt(nameloss_co, delimiter = ','
                    , dtype = float)
    lossc = np.loadtxt(nameloss_lo, delimiter = ','
                    , dtype = float)
    epok = np.loadtxt(nameepoc_co, delimiter = ','
                    , dtype = float)
    epoc = np.loadtxt(nameepoc_lo, delimiter = ','
                    , dtype = float)
    
    effk = effk.mean(axis = 0)
    effc = effc.mean(axis = 0)
    lossk = lossk.mean(axis = 0)
    lossc = lossc.mean(axis = 0)
    epok = epok.mean(axis = 0)
    epoc = epoc.mean(axis = 0)

    delta_y = effk - effc

    fig, axs = plt.subplots(2,figsize = (12,8))
    axs[0].plot(data, effk, 'o-', color = 'green', label = 'Cosine Squasher')
    axs[0].plot(data, effc, 'o-', color = 'red', label = 'Logistic Function')
    if data == lr:
        axs[0].set_xlabel('Learning Rate')
    elif data == dimension:
        axs[0].set_xlabel('Dimensione dei punti')
    else:
        axs[0].set_xlabel('Numero di neuroni nello strato nascosto')
    axs[0].set_ylabel('Efficienza (%)')
    axs[0].grid(True)
    axs[0].legend()
    axs[1].plot(data, delta_y/(delta_y+effc)*100, 'o-', color = 'orange', label = 'Differenza relativa %')
    if data == lr:
        axs[1].set_xlabel('Learning Rate')
    elif data == dimension:
        axs[1].set_xlabel('Dimensione dei punti')
    else:
        axs[1].set_xlabel('Numero di neuroni nello strato nascosto')
    axs[1].set_ylabel('Differenza relativa (%)')
    axs[1].grid(True)
    axs[1].legend()
    axs[1].set_yscale('log')
    axs[1].set_ylim([0, 100])
    plt.tight_layout()
    plt.savefig('IMG\\'+name_eff+'.png')

    fig, axs = plt.subplots(figsize = (12,8))
    axs.plot(data, lossk, 'o-', color = 'green', label = 'Cosine Squasher')
    axs.plot(data, lossc, 'o-', color = 'red', label = 'Logistic Function')
    if data == lr:
        axs.set_xlabel('Learning Rate')
    elif data == dimension:
        axs.set_xlabel('Dimensione dei punti')
    else:
        axs.set_xlabel('Numero di neuroni nello strato nascosto')
    axs.set_ylabel('Loss function')
    axs.grid(True)
    axs.legend()
    axs.set_yscale('log')
    plt.tight_layout()
    plt.savefig('IMG\\'+name_loss+'.png')

    fig, axs = plt.subplots(figsize = (12,8))
    axs.plot(data, epok, 'o-', color = 'green', label = 'Cosine Squasher')
    axs.plot(data, epoc, 'o-', color = 'red', label = 'Logistic Function')
    if data == lr:
        axs.set_xlabel('Learning Rate')
    elif data == dimension:
        axs.set_xlabel('Dimensione dei punti')
    else:
        axs.set_xlabel('Numero di neuroni nello strato nascosto')
    axs.set_ylabel('Numero di epoche')
    axs.grid(True)
    axs.legend()
    plt.tight_layout()
    plt.savefig('IMG\\'+name_epoc+'.png')
