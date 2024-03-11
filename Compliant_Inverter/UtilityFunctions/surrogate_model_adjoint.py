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


class SurrogateModel_adjoint():
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
        self.XDBC1 = tf.convert_to_tensor(data['DBC1'],dtype=datatype)
        self.XDBC2 = tf.convert_to_tensor(data['DBC2'],dtype=datatype)
        self.resample()
        np.savetxt('train_set.txt',np.array(self.trainset))
        gc.collect()

    def nonzero_region_sampling(self,N,L,H):
        samples = []
        while len(samples)<N:
            x = L * np.random.rand()
            y = H * np.random.rand() 
            p = griddata(self.mesh['Xint'].reshape((-1,2)), self.data['rho_quad'].reshape((-1,1)), (x, y), method='nearest', fill_value=0.)
            if p>1e-3:
                samples.append((x, y))
        return samples


    def resample(self):
        # print('resampling')
        # X_re = np.array(self.nonzero_region_sampling(10000,self.mesh['L'],self.mesh['H']))
        X_re = np.zeros((self.mesh['Nsample'],2))
        X_re[:,0:1] = np.random.rand(self.mesh['Nsample'],1)*self.mesh['L']
        X_re[:,1:2] = np.random.rand(self.mesh['Nsample'],1)*self.mesh['H']
        rho_re = griddata(self.mesh['Xint'].reshape((-1,2)), self.data['rho_quad'].reshape((-1,1)), (X_re[:,0], X_re[:,1]), method='nearest', fill_value=0.)
        # ut.plotresult_contour('./','test',X_re,rho_re,L=1,H=1)
        self.trainset = np.concatenate((X_re,rho_re),axis=1)
        self.trainset = tf.convert_to_tensor(self.trainset,dtype=datatype)
        self.train_ds = tf.data.Dataset.from_tensor_slices((self.trainset))
        self.train_ds = self.train_ds.shuffle(40000).batch(self.trainParams['Nbatch'])
        gc.collect()

    def initialize_model(self):
        self.Model = ut.create_NN_model(self.trainParams,mean=self.mean,var=self.var)
        if os.path.isfile('model_ad_0.h5'):
            self.Model.load_weights('model_ad_0.h5')
        self.loss_tracker = ut.LossTracking()
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.trainParams['lr'])
        self.Model.compile(optimizer=optimizer)
        _callbacks = [tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=30)]
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
                if epoch%50==0:
                    self.loss_tracker.print()
            self.loss_tracker.reset()

            # Callback at the end of epoch
            self.callbacks.on_epoch_end(epoch, logs={'total_loss': total_loss})

            stopEarly = self.Callback_EarlyStopping(self.loss_tracker.loss_history['total_loss'], min_delta=1e-8, patience=20)
            if stopEarly:
                print("Callback_EarlyStopping signal received at epoch= %d"%(epoch))
                print("Terminating training ",'\n')
                break 
            # Re-shuffle dataset
            self.train_ds = tf.data.Dataset.from_tensor_slices((self.trainset))
            self.train_ds = self.train_ds.shuffle(40000).batch(self.trainParams['Nbatch']) 
            # if epoch%10== 0:
            #     U_pinn,V_pinn = self.Model.predict(self.mesh['input_dof'])
            #     ut.plotresult_contour('prediction/','solution_lmU.png',self.mesh['input_dof'],U_pinn,L=120,H=60,cmap='seismic')
            #     ut.plotresult_contour('prediction/','solution_lmV.png',self.mesh['input_dof'],V_pinn,L=120,H=60,cmap='seismic')
        self.Endepoch = epoch

    @tf.function()
    def SIMP_E(self,rhoi):
        return self.material['Emin']+(self.material['E']-self.material['Emin'])*rhoi**3

    @tf.function()
    def grad_SIMP_E(self,rhoi):
        return 3*(self.material['E']-self.material['Emin'])*rhoi**2  

    @tf.function()
    def eps_rho(self,x,y):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            u, v = self.Model(tf.stack([x, y],axis=1))
        udx = tape.gradient(u,x)
        udy = tape.gradient(u,y)        
        vdx = tape.gradient(v,x)
        vdy = tape.gradient(v,y)
        epsxx = udx
        epsxy = 1/2*(udy + vdx)
        epsyy = vdy
        del tape
        return epsxx,epsxy,epsyy
    
    def nabla_psi(self,x,y):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            u, v = self.Model(tf.stack([x, y],axis=1))
        udx = tape.gradient(u,x)
        udy = tape.gradient(u,y)        
        vdx = tape.gradient(v,x)
        vdy = tape.gradient(v,y)
        return udx,udy,vdx,vdy

    @tf.function()
    def treps(self,epsxx,epsyy):
        return epsxx + epsyy

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
    
    @tf.function()
    def sigma_rho(self,x,y,rho_res):
        epsxx,epsxy,epsyy = self.eps_rho(x,y)
        treps = epsxx+epsyy
        simpe = self.SIMP_E(rho_res)
        # print(np.shape(epsxx))
        sigxx = self.mul2(simpe*self.material['lambda'],treps)+self.mul2(2*self.material['mu']*tf.squeeze(epsxx),simpe)
        sigxy = self.mul2(2*simpe*self.material['mu'],epsxy)
        sigyy = self.mul2(simpe*self.material['lambda'],treps)+self.mul2(2*self.material['mu']*tf.squeeze(epsyy),simpe)
        # sigxx = self.simpe[:]*self.lbda*treps+2*self.mu*epsxx
        # sigxy = self.simpe[:]*2*self.mu*epsxy
        # sigyy = self.simpe[:]*self.lbda*treps+2*self.mu*epsyy
        # print(np.shape(sigxx))
        # ut.plotresult('surrogate_result/','simpe.png',self.input_in,self.simpe)
        return sigxx, sigxy, sigyy 
    
    @tf.function()
    def Energy_in(self,x,y,rho_res):
        epsxx,epsxy,epsyy = self.eps_rho(x,y)
        sigxx,sigxy,sigyy = self.sigma_rho(x,y,rho_res)
        Ein = 120.*60.*0.5*(self.mul2(sigxx,epsxx)+self.mul2(2*sigxy,epsxy)+self.mul2(sigyy,epsyy))
        # Ein = np.size(Wint)*Wint*0.5*(self.mul2(sigxx,epsxx)+self.mul2(2*sigxy,epsxy)+self.mul2(sigyy,epsyy))
        # res=0.5*(sigxx*epsxx+2*sigxy*epsxy+sigyy*epsyy)*self.W_in[:,None]
        # print(np.shape(epsxx))
        return Ein
    
    @tf.function()
    def Energy_ex(self):
        uin,_ = self.Model(np.array([[0.],[60.]]).T)
        uout,_ = self.Model(np.array([[120.],[60.]]).T)
        Eex = tf.squeeze(uout*1.)-tf.squeeze(uin*(0.5*1.*uin))-tf.squeeze(0.5*uout*(0.001)*uout)
        return Eex

    @tf.function()
    def data_loss_calculator(self):
        u1,v1 = self.Model(self.XDBC1)
        u2,v2 = self.Model(self.XDBC2)
        loss_DBC1 = tf.reduce_mean(tf.square(tf.squeeze(u1)))+tf.reduce_mean(tf.square(tf.squeeze(v1)))
        loss_DBC2 = tf.reduce_mean(tf.square(tf.squeeze(v2)))
        loss = loss_DBC1+loss_DBC2
        return loss
    # def data_loss_calculator(self,Xi,Ui,Vi):
    #     u,v = self.Model(Xi)
    #     loss_x = tf.reduce_mean(tf.abs((tf.squeeze(u)-tf.squeeze(Ui))))
    #     loss_y = tf.reduce_mean(tf.abs((tf.squeeze(v)-tf.squeeze(Vi))))
    #     loss = loss_x+loss_y
    #     return loss
    
    @tf.function()
    def pde_loss_calculator(self,xi,yi,rhoi):
        Ein = self.Energy_in(xi,yi,rhoi)
        Eex = self.Energy_ex()
        Ep = tf.reduce_mean(Ein-Eex)
        return  Ep
    
    @tf.function()
    def train_step(self,Xi):
        with tf.GradientTape() as tape:
            tape.watch(self.Model.trainable_weights)
            # data_loss = self.data_loss_calculator(Xi[:,:2],Xi[:,-2],Xi[:,-1])
            data_loss = self.data_loss_calculator()
            phys_loss = self.pde_loss_calculator(Xi[:,0:1],Xi[:,1:2],Xi[:,2:3])
            total_loss = 1.*data_loss+1*phys_loss
        gradients = tape.gradient(total_loss,self.Model.trainable_variables)
        return total_loss, data_loss, phys_loss, gradients
    
    def Callback_EarlyStopping(self, LossList, min_delta=1e-8, patience=50):
        #No early stopping for 2*patience epochs 
        if len(LossList)//patience < 2 :
            return False
        #Mean loss for last patience epochs and second-last patience epochs
        mean_previous = np.mean(LossList[::-1][patience:2*patience]) #second-last
        mean_recent = np.mean(LossList[::-1][:patience]) #last
        #you can use relative or absolute change
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
        plt.savefig('surrogate_result/history_ad.png')
