from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gc
import os
from scipy.interpolate import griddata

import utils as ut

tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
tol=1e-6
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
datatype = 'float32'


class SurrogateModel_primal():
    def __init__(self) -> None:
        pass
    
    def updatemesh(self,mesh):
        self.mesh = mesh

    def updatematerial(self,material):
        self.material = material

    def updatetrainParams(self,trainParams):
        self.trainParams = trainParams

    def updatetrainset(self,data):
        self.data = data
        self.mean = np.mean(self.mesh['input_dof'],axis=0)
        self.var = np.var(self.mesh['input_dof'],axis=0)
        self.XDBC = tf.convert_to_tensor(self.data['DBC'],dtype=datatype)
        self.resample()

    def resample(self):
        # print('resampling')
        X_re = np.zeros((self.mesh['Nsample'],2))
        X_re[:,0:1] = np.random.rand(self.mesh['Nsample'],1)*self.mesh['L']
        X_re[:,1:2] = np.random.rand(self.mesh['Nsample'],1)*self.mesh['H']
        rho_re = griddata(self.mesh['Xint'].reshape((-1,2)), self.data['rho_quad'].reshape((-1,1)), (X_re[:,0], X_re[:,1]), method='nearest', fill_value=0.)
        self.trainset = np.concatenate((X_re,rho_re),axis=1)
        self.trainset = tf.convert_to_tensor(self.trainset,dtype=datatype)
        self.train_ds = tf.data.Dataset.from_tensor_slices((self.trainset))
        self.train_ds = self.train_ds.shuffle(40000).batch(self.trainParams['Nbatch'])
        gc.collect()
        

    def initialize_model(self):
        self.Model = ut.create_NN_model(self.trainParams,mean=self.mean,var=self.var)
        if os.path.isfile('model_pr_0.h5'):
            self.Model.load_weights('model_pr_0.h5')
            print('loading initial weights...')
        self.loss_tracker = ut.LossTracking()
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.trainParams['lr'])
        self.Model.compile(optimizer=optimizer)
        _callbacks = [tf.keras.callbacks.ReduceLROnPlateau(factor=0.5,min_delta=1e-6),
             tf.keras.callbacks.ModelCheckpoint('DeepONet_model.h5', monitor='total_loss', save_best_only=True)]
        self.callbacks = tf.keras.callbacks.CallbackList(
            _callbacks, add_history=False, model=self.Model)
        
    def trainmodel(self,verbose=False):
        for epoch in range(1,self.trainParams['Nepoch']+1):
            if epoch%10==0:
                self.resample()
            if verbose==True:
                if epoch%50==0:
                    print(f"Epoch {epoch}:")
                    print(f"lr: {self.Model.optimizer.lr.numpy():.2e}")
            for X in self.train_ds:
                # Calculate gradients,
                total_loss, data_loss, phys_loss, gradients = self.train_step(X)               
                # Gradient descent
                self.Model.optimizer.apply_gradients(zip(gradients, self.Model.trainable_variables))
                # Loss tracking
                self.loss_tracker.update(total_loss, data_loss, phys_loss)
            # Loss summary
            self.loss_tracker.history()
            if verbose==True:
                if epoch%50 ==0:
                    self.loss_tracker.print()
            self.loss_tracker.reset()

            # Callback at the end of epoch
            self.callbacks.on_epoch_end(epoch, logs={'total_loss': total_loss})

            stopEarly = self.Callback_EarlyStopping(self.loss_tracker.loss_history['total_loss'], min_delta=1e-8, patience=30)
            if stopEarly:
                print("Callback_EarlyStopping signal received at epoch= %d"%(epoch))
                print("Terminating training ",'\n')
                break 

        self.Endepoch = epoch

    @tf.function()
    def SIMP_k(self,rhoi):
        return self.material['kmin']+(self.material['k']-self.material['kmin'])*rhoi**3

    @tf.function()
    def grad_SIMP_k(self,rhoi):
        return 3*(self.material['k']-self.material['kmin'])*rhoi**2  

    @tf.function()
    def mul3(self,ele1,ele2,ele3):
        ele1=tf.squeeze(ele1)
        ele2=tf.squeeze(ele2)
        ele3=tf.squeeze(ele3)
        return tf.multiply(tf.multiply(ele1,ele2),ele3)
    
    @tf.function()
    def mul2(self,ele1,ele2):
        ele1=tf.squeeze(ele1)
        ele2=tf.squeeze(ele2)
        return tf.multiply(ele1,ele2)
    
    def nabla_T(self,x,y):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            t = self.Model(tf.stack([x, y],axis=1))
        tdx = tape.gradient(t,x)
        tdy = tape.gradient(t,y) 
        del tape
        return tdx,tdy
    
    @tf.function()
    def Energy(self,x,y,rho_res):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            t = self.Model(tf.stack([x, y],axis=1))
        tdx = tape.gradient(t,x)
        tdy = tape.gradient(t,y) 
        del tape
        simpk = self.SIMP_k(rho_res)
        E = self.mesh['L']*self.mesh['H']*(1/2*self.mul2(simpk,(tdx**2+tdy**2))-tf.squeeze(t)*self.material['s'])
        return E

    @tf.function()
    def data_loss_calculator(self):
        t = self.Model(self.XDBC)
        loss_DBC = tf.reduce_mean(tf.square(tf.squeeze(t)))
        return loss_DBC
    
    @tf.function()
    def pde_loss_calculator(self,xi,yi,rhoi):
        E = self.Energy(xi,yi,rhoi)
        Ep = tf.reduce_mean(E)
        return  Ep
    
    @tf.function()
    def train_step(self,Xi):
        with tf.GradientTape() as tape:
            tape.watch(self.Model.trainable_weights)
            # data_loss = self.data_loss_calculator(Xi[:,:2],Xi[:,-2],Xi[:,-1])
            data_loss = self.data_loss_calculator()
            phys_loss = self.pde_loss_calculator(Xi[:,0:1],Xi[:,1:2],Xi[:,2:3])
            total_loss = 1.*data_loss+1.*phys_loss
        gradients = tape.gradient(total_loss,self.Model.trainable_variables)
        return total_loss, data_loss, phys_loss, gradients
    
    def Callback_EarlyStopping(self, LossList, min_delta=1e-4, patience=60):
        #No early stopping for 2*patience epochs 
        if len(LossList)//patience < 2 :
            return False
        #Mean loss for last patience epochs and second-last patience epochs
        mean_previous = np.mean(LossList[::-1][patience:2*patience]) #second-last
        mean_recent = np.mean(LossList[::-1][:patience]) #last
        delta_abs = np.abs(mean_recent - mean_previous) #abs change
        delta_abs = np.abs(delta_abs / mean_previous)  # relative change
        if delta_abs < min_delta :
            print("*CB_ES* Loss didn't change much from last %d epochs"%(patience))
            print("*CB_ES* Percent change in loss value:", delta_abs*1e2)
            return True
        else:
            return False
    

    def plothist(self):
        # History
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].plot(range(len(self.loss_tracker.loss_history['data_loss'])), self.loss_tracker.loss_history['data_loss'])
        ax[1].plot(range(len(self.loss_tracker.loss_history['PDE_loss'])), np.abs(self.loss_tracker.loss_history['PDE_loss']))
        ax[2].plot(range(len(self.loss_tracker.loss_history['total_loss'])), np.abs(self.loss_tracker.loss_history['total_loss']))
        ax[0].set_title('Data Loss')
        ax[1].set_title('PDE Loss')
        ax[2].set_title('Total Loss')
        for axs in ax:
            axs.set_yscale('log')
        plt.savefig('history.png')
