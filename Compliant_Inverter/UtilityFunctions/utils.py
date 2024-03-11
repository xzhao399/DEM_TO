from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib
# from matplotlib.mlab import griddata
from scipy.interpolate import griddata
from scipy.sparse import coo_matrix
import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential,Model
from tf_fourier_features import FourierFeatureProjection
import os
from collections import defaultdict

from Geom_examples import Quadrilateral

np.random.seed(42)
tf.random.set_seed(42)

dataType='float32'
plt.rcParams.update({'font.size': 22})


####################################################################
#                 funtions for calculating errors                     #
####################################################################

def calculate_error_rL2(array1,array2):
    if array1.shape != array2.shape:
        print("Warning: Arrays have different shapes. Exiting.")
        raise SystemExit  # Stop the code execution
    # Calculate the difference of the arrays
    error = np.linlg.norm(array1-array2)/np.linalg.norm(array1) #root mean square error
    return error


def calculate_error_RMSE(array1,array2):
    if array1.shape != array2.shape:
        print("Warning: Arrays have different shapes. Exiting.")
        raise SystemExit  # Stop the code execution
    # Calculate the difference of the arrays
    error = np.sqrt(np.mean((array1-array2)**2)) #root mean square error
    return error

def calculate_error_MSE(array1,array2):
    if array1.shape != array2.shape:
        print("Warning: Arrays have different shapes. Exiting.")
        raise SystemExit  # Stop the code execution
    # Calculate the difference of the arrays
    error = (np.mean((array1-array2)**2)) #root mean square error
    return error

def calculate_error_re(array1,array2):
    if array1.shape != array2.shape:
        print("Warning: Arrays have different shapes. Exiting.")
        raise SystemExit  # Stop the code execution
    # Calculate the difference of the arrays)
    error = np.mean(np.abs(array1-array2)/(np.abs(array1)+1e-10)) 
    return error

def calculate_error_MAE(array1,array2):
    if array1.shape != array2.shape:
        print("Warning: Arrays have different shapes. Exiting.")
        raise SystemExit  # Stop the code execution
    # Calculate the difference of the arrays
    error = np.mean(np.abs(array1-array2)) #root mean square error
    return error

def cosine_similarity(U,V):
    U = np.squeeze(U)
    V = np.squeeze(V)
    if U.shape != V.shape:
        print("Warning: Arrays have different shapes. Exiting.")
        raise SystemExit  # Stop the code execution
    aod = np.sum(U*V)/(np.linalg.norm(U)*np.linalg.norm(V))
    return aod


####################################################################
#                 funtions for meshing                     #
####################################################################

def findeleinx(Nxfem, Nyfem, hx, hy, x, y):
    elx = int(x // hx)
    ely = int((Nyfem * hy - y) // hy)  
    elx = min(max(elx, 0), Nxfem - 1)
    ely = min(max(ely, 0), Nyfem - 1)
    el = ely + elx * Nyfem

    return el
    

def find_ele_inx_array(Nelmufem, Nelmvfem, L, H, x, y):

    hx = L / Nelmufem
    hy = H / Nelmvfem
    
    elx = np.floor(x / hx).astype(int)
    ely = np.floor((H - y) / hy).astype(int)  

    elx = np.clip(elx, 0, Nelmufem - 1)
    ely = np.clip(ely, 0, Nelmvfem - 1)
    
    idx = ely + elx * Nelmvfem

    return idx

def getquadpts(L=2.,H=1.,Nelmufem=40,Nelmvfem=20,numElemU=80,numElemV=40,numGauss=2,bdcode=1):
    domainCorners = np.array([[0., 0.], [0, H], [L, 0.], [L, H]])
    geomDomain = Quadrilateral(domainCorners)
    xPhys, yPhys, Wint = geomDomain.getQuadIntPts(numElemU, numElemV, numGauss) 
    xPhysBnd, yPhysBnd, xNorm, yNorm, Wbnd = geomDomain.getQuadEdgePts(numElemV, numGauss, bdcode)
    Xint = np.concatenate((xPhys,yPhys),axis=1).astype(dataType)
    Xbnd = np.concatenate((xPhysBnd, yPhysBnd, xNorm, yNorm), axis=1).astype(dataType)
    eleidx = find_ele_inx_array(Nelmufem,Nelmvfem,L,H,xPhys,yPhys)  
    eleidx = np.squeeze(np.int_(eleidx) )
    return Xint, Wint, Xbnd, Wbnd, eleidx

def projrho(density,idx):
    return density[idx]

def inversesort(inverseidx,design,design_indexed):
    design[inverseidx] = design_indexed
    return design

def generatedofcoord(nelx=80, nely=40, L=2., H=1.):
    dx = L / nelx
    dy = H / nely
    x_coords = np.arange(0, L + dx, dx)  
    y_coords = np.arange(H, -dy, -dy)   

    X_dof = np.empty(((nelx + 1) * (nely + 1), 2))  
    X_dof[:, 0] = np.repeat(x_coords, nely + 1)  
    X_dof[:, 1] = np.tile(y_coords, nelx + 1)   
    evenidx = np.arange((nelx + 1) * (nely + 1)) * 2
    oddidx = evenidx + 1

    return X_dof, evenidx, oddidx
    

####################################################################
#                 funtions for path                     #
####################################################################
def getpath(foldername):
    directory = foldername
    parent_dir=os.getcwd()
    path = os.path.join(parent_dir, directory)  
    path = path+'/'   
    return path

def makepath(foldername):
    directory = foldername
    parent_dir=os.getcwd()
    path = os.path.join(parent_dir, directory)
    if not os.path.exists(path):
        os.makedirs(path)
    path = path+'/' # output folder   
    return path

####################################################################
#                 Initialize Density filter                      #
####################################################################    
def DensityFilter_init(rmin,nelx,nely):  
    nfilter=int(nelx*nely*((2*(np.ceil(rmin)-1)+1)**2))
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    cc=0
    for i in range(nelx):
        for j in range(nely):
            row=i*nely+j
            kk1=int(np.maximum(i-(np.ceil(rmin)-1),0))
            kk2=int(np.minimum(i+np.ceil(rmin),nelx))
            ll1=int(np.maximum(j-(np.ceil(rmin)-1),0))
            ll2=int(np.minimum(j+np.ceil(rmin),nely))
            for k in range(kk1,kk2):
                for l in range(ll1,ll2):
                    col=k*nely+l
                    fac=rmin-np.sqrt(((i-k)*(i-k)+(j-l)*(j-l)))
                    iH[cc]=row
                    jH[cc]=col
                    sH[cc]=np.maximum(0.0,fac)
                    cc=cc+1
    # Finalize assembly and convert to csc format
    H=coo_matrix((sH,(iH,jH)),shape=(nelx*nely,nelx*nely)).tocsc()
    Hs=H.sum(1)
    return H,Hs


####################################################################
#                 funtions for plotting                     #
####################################################################
def comparewith_fem_sln(coord,U,V):
    U = squeeze(U)
    V = squeeze(V)
    # coord = np.loadtxt('coord.txt')
    ux_fem = squeeze(np.loadtxt('reference_result/ux.txt'))
    uy_fem = squeeze(np.loadtxt('reference_result/uy.txt'))
    er = (linalg.norm(U-ux_fem)**2+linalg.norm(V-uy_fem)**2)/(linalg.norm(ux_fem)**2+linalg.norm(uy_fem)**2)
    er = np.sqrt(er)
    er_u = linalg.norm(U-ux_fem)/linalg.norm(ux_fem)
    er_v = linalg.norm(V-uy_fem)/linalg.norm(uy_fem)
    print('The total error is', er)
    plt.figure()
    plt.scatter(coord[:,0],coord[:,1],5,abs(U-ux_fem),cmap='jet')
    plt.colorbar()
    plt.xlim([0,2])
    plt.ylim([0,1])
    plt.axis('equal')
    plt.title('u disp abs er, L2 relative error = %f'%(er_u))
    plt.savefig('VPINN_result/u_wither',bbox_inches='tight')
    plt.close('all')

    plt.figure()
    plt.scatter(coord[:,0],coord[:,1],5,abs(V-uy_fem),cmap='jet')
    plt.colorbar()
    plt.xlim([0,2])
    plt.ylim([0,1])
    plt.axis('equal')
    plt.title('v disp abs er, L2 relative error = %f'%(er_v))
    plt.savefig('VPINN_result/v_wither',bbox_inches='tight')
    plt.close('all')

    return er_u,er_v

def plotdesign(path,figname,xPhys,nelx=80,nely=20,vmin=0,vmax=1):
    fig,ax = plt.subplots()
    xPhys = np.array(xPhys)
    im = ax.imshow(xPhys.reshape((nelx,nely)).T, cmap='gray_r',\
    interpolation='none',norm=colors.Normalize(vmin=vmin,vmax=vmax))
    # fig.colorbar(im,ax=ax,fraction=0.025)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    fig.show()
    fig.canvas.draw()
    plt.savefig(path+figname,bbox_inches='tight')
    plt.close('all')

def plotresult(path,figname,X,U,cmap='jet',L=4,H=1):
    fig=plt.figure()
    ax = fig.add_subplot(111)
    # ax = fig.add_subplot(111, projection='3d')
    # im=ax.scatter(x,y,T,c=T)#, marker='o')
    # x,y,z = grid(np.squeeze(X),U)
    # im = ax.contourf(x,y,z,cmap=cmap)
    U = squeeze(U)
    im = ax.scatter(X[:,0],X[:,1],s=10,c=U,cmap=cmap,norm=None)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title(figname)
    ax.set_xlim(0,L)
    ax.set_ylim(0,H)
    ax.set_aspect('equal')
    fig.colorbar(im,ax=ax,fraction=0.025, norm=norm)
    plt.savefig(path+figname+".png",bbox_inches='tight')
    plt.clf()
    plt.close('all')

def plotresult_contour(path,figname,X,U,cmap='jet',L=4,H=1,vmin=None,vmax=None,norm=None,nelx=80,nely=20,title='figname'):
    fig=plt.figure()
    ax = fig.add_subplot(111)
    u = squeeze(U)
    x = X[:,0]
    y = X[:,1]
    x_range = np.linspace(min(x), max(x), num=100)
    y_range = np.linspace(min(y), max(y), num=100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = griddata((x, y), u, (X, Y), method='linear')
    im = ax.pcolor(X, Y, Z,cmap=cmap,vmin=vmin,vmax = vmax, norm=norm)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    if title == 'figname':
        ax.set_title(figname)
    ax.set_xlim(0,L)
    ax.set_ylim(0,H)
    ax.set_aspect('equal')
    fig.colorbar(im,ax=ax,fraction=0.025)
    plt.savefig(path+figname+".png",bbox_inches='tight')
    plt.clf()
    plt.close('all')

def plotcurve(path,name,hist):
    plt.figure()
    plt.plot(hist)
    plt.xlabel('iter')
    plt.ylabel(name)
    plt.savefig(path+name+'.png',bbox_inches='tight')
    plt.close('all')

def plotresult1D(path,name,x,h1):
    plt.figure()
    plt.plot(x,h1)
    plt.xlabel('x')
    plt.ylabel(name)
    plt.savefig(path+name+'.png',bbox_inches='tight')
    plt.close('all') 


def Set_variables(x,variables):
    idx = 0
    for v in variables:
        vs=v.shape
        if len(vs)==4:
            sw=vs[0]*vs[1]*vs[2]*vs[3]
            new_val=tf.reshape(x[idx:idx+sw],(vs[0],vs[1],vs[2],vs[3]))
            idx+=sw
        elif len(vs)==3:
            sw=vs[0]*vs[1]*vs[2]
            new_val=tf.reshape(x[idx:idx+sw],(vs[0],vs[1],vs[2]))
            idx+=sw
        elif len(vs)==2:  
            sw=vs[0]*vs[1]
            new_val=tf.reshape(x[idx:idx+sw],(vs[0],vs[1]))
            idx+=sw
        elif len(vs)==1:
            new_val=x[idx:idx+vs[0]]
            idx+=vs[0]
        elif len(vs)==0:
            new_val=x[idx]
            idx+=1
        v.assign(tf.cast(new_val,tf.float64))
        # v.assign(new_val)
    return None

def plotpoints(path,figname,X,L,H):
    fig=plt.figure()
    ax = fig.add_subplot(111)
    # ax = fig.add_subplot(111, projection='3d')
    # im=ax.scatter(x,y,T,c=T)#, marker='o')
    im = ax.scatter(X[:,0],X[:,1],s=5,c='red')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_xlim(0,L)
    ax.set_ylim(0,H)
    ax.set_aspect('equal')
    fig.colorbar(im,ax=ax)
    plt.savefig(path+figname+".png",bbox_inches='tight')
    plt.clf()

####################################################################
#                 funtions for NN model                            #
####################################################################

def create_NN_model(trainParams,mean=None,var=None,verbose=False):
    input = tf.keras.Input(shape=(trainParams['N_input_coord'],), name="coord")  
    x = tf.keras.layers.Normalization(mean=mean, variance=var)(input)

    def NN_block(t):
        initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=trainParams['rff_dev'])
        # initializer2 = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-5)
        t = 2.0*np.pi*t
        t = tf.keras.layers.Dense(trainParams['Nneuron']//2, use_bias=False, trainable=True,kernel_initializer=initializer)(t)
        t = tf.concat([tf.sin(t),tf.cos(t)],axis=1)
        for i in range(trainParams['Nlayers']):
            t = tf.keras.layers.Dense(trainParams['Nneuron'], activation=trainParams['activation'],kernel_initializer='glorot_normal')(t)
        t = tf.keras.layers.Dense(1)(t)
        return t
    
    u_output = NN_block(x)*(tf.math.maximum(0.,input[:,1:2]-60/30)/60+input[:,0:1]/120)
    v_output = NN_block(x)*(tf.math.maximum(0.,input[:,1:2]-60/30)/60+input[:,0:1]/120)*(60.-input[:,1:2])/60

    model = tf.keras.models.Model(inputs=input,outputs=[u_output,v_output])
    if verbose:
        model.summary()
    return model

class LossTracking:
    def __init__(self):
        self.mean_total_loss = tf.keras.metrics.Mean()
        self.mean_data_loss = tf.keras.metrics.Mean()
        self.mean_PDE_loss = tf.keras.metrics.Mean()
        self.loss_history = defaultdict(list)

    def update(self, total_loss, data_loss, PDE_loss):
        self.mean_total_loss(total_loss)
        self.mean_data_loss(data_loss)
        self.mean_PDE_loss(PDE_loss)

    def reset(self):
        self.mean_total_loss.reset_states()
        self.mean_data_loss.reset_states()
        self.mean_PDE_loss.reset_states()

    def print(self):
        print(f"Data={self.mean_data_loss.result().numpy():.4e}, \
              PDE={self.mean_PDE_loss.result().numpy():.4e}, \
              total_loss={self.mean_total_loss.result().numpy():.4e}")
        
    def history(self):
        self.loss_history['total_loss'].append(self.mean_total_loss.result().numpy())
        self.loss_history['data_loss'].append(self.mean_data_loss.result().numpy())
        self.loss_history['PDE_loss'].append(self.mean_PDE_loss.result().numpy())


def preprocess(x):
    x = tf.cast(x, dtype='float32')
    return x


