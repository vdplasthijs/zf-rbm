
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 11:22:02 2016

Probabilistic Graphical Model (PGM) Class.

A general class for any kind of PGM. Has a 'gen_data' method.


TO DO

Methods :
- gen_data
- markov_step

@author: jerometubiana
"""



import numpy as np
import utilities7 as utilities
import copy
from scipy.sparse import diags
import time
import layer7 as layer


double_precision = False

if double_precision:
    curr_float = np.float64
    curr_int = np.int64
else:
    curr_float = np.float32
    curr_int = np.int16


def couplings_gradients(W,X1_p,X1_n, X2_p,X2_n, n_c1, n_c2,mean1= False, mean2= False, l1 = 0, l1b = 0, l1c= 0, l2 = 0,l1_custom = None,l1b_custom=None,weights=None,weights_neg=None):
    update =utilities.average_product(X1_p, X2_p, c1 = n_c1,c2=n_c2, mean1 = mean1, mean2= mean2,weights=weights) - utilities.average_product(X1_n, X2_n, c1 = n_c1,c2=n_c2, mean1 = mean1, mean2= mean2,weights=weights_neg)
    if l2>0:
        update -= l2 * W
    if l1>0:
        update -= l1 * np.sign( W)
    if l1b>0: # NOT SUPPORTED FOR POTTS
        if n_c2 > 1: # Potts RBM.
            update -= l1b * np.sign(W) *  np.abs(W).mean(-1).mean(-1)[:,np.newaxis,np.newaxis]
        else:
            update -= l1b * np.sign( W) * (np.abs(W).sum(1))[:,np.newaxis]
    if l1c>0: # NOT SUPPORTED FOR POTTS
        update -= l1c * np.sign( W) * ((np.abs(W).sum(1))**2)[:,np.newaxis]
    if l1_custom is not None:
        update -= l1_custom * np.sign(W)
    if l1b_custom is not None:
        update -= l1b_custom[0] * (l1b_custom[1]* np.abs(W)).mean(-1).mean(-1)[:,np.newaxis,np.newaxis] *np.sign(W)

    if weights is not None:
        update *= weights.sum()/X1_p.shape[0]
    return update


def couplings_gradients_h(W,X1_p,X1_n, X2_p,X2_n, n_c1, n_c2, l1 = 0, l1b = 0, l1c= 0, l2 = 0,l1_custom = None,l1b_custom=None,weights=None,weights_neg=None):
    update =utilities.average_product(X1_p, X2_p, c1 = n_c1,c2=n_c2, mean1 = True, mean2= False,weights=weights) - utilities.average_product(X1_n, X2_n, c1 = n_c1,c2=n_c2, mean1 = False, mean2= True,weights=weights_neg)
    if l2>0:
        update -= l2 * W
    if l1>0:
        update -= l1 * np.sign( W)
    if l1b>0: # NOT SUPPORTED FOR POTTS
        if n_c2 > 1: # Potts RBM.
            update -= l1b * np.sign(W) *  np.abs(W).mean(-1).mean(-1)[:,np.newaxis,np.newaxis]
        else:
            update -= l1b * np.sign( W) * (np.abs(W).sum(1))[:,np.newaxis]
    if l1c>0: # NOT SUPPORTED FOR POTTS
        update -= l1c * np.sign( W) * ((np.abs(W).sum(1))**2)[:,np.newaxis]
    if l1_custom is not None:
        update -= l1_custom * np.sign(W)
    if l1b_custom is not None:
        update -= l1b_custom[0] * (l1b_custom[1]* np.abs(W)).mean(-1).mean(-1)[:,np.newaxis,np.newaxis] *np.sign(W)

    if weights is not None:
        update *= weights.sum()/X1_p.shape[0]
    return update




def gauge_adjust_couplings(W,n_c1,n_c2,gauge='zerosum'):
    if gauge == 'zerosum':
        if (n_c1 >1) & (n_c2 >1):
            W =W # To be changed...
        elif (n_c1 ==1) & (n_c2 >1):
            W-= W.sum(-1)[:,:,np.newaxis]/n_c2
        elif (n_c1 >1) & (n_c2 ==1):
            W-= W.sum(-1)[:,:,np.newaxis]/n_c1
    else:
        print('adjust_couplings -> gauge not supported')
    return W



class PGM(object):
    def __init__(self, n_layers = 3, layers_name =['layer1','layer2','layer3'], layers_size = [100,20,30],layers_nature = ['Bernoulli','Bernoulli','Bernoulli'], layers_n_c = [None,None,None]):
        self.n_layers = n_layers
        self.layers_name = layers_name
        self.layers_size = layers_size
        self.layers_nature = layers_nature
        self.layers_n_c = layers_n_c


    def markov_step(self,config,beta =1):
        return config

    def markov_step_PT(self,config):
        for i,beta in zip(np.arange(self.N_PT), self.betas):
            config[i] = self.markov_step(config[i],beta =beta)
        return config

    def markov_step_and_energy(self, config,E, beta=1):
        return config,E


    def gen_data(self, Nchains = 10, Lchains = 100, Nthermalize = 0 ,Nstep = 1, N_PT =1, config_init = [], beta = 1,batches = None,reshape = True,record_replica = False, record_acceptance=None, update_betas = None,update_betas_lr = 0.1,update_betas_lr_decay=0.99,update_MoI_lr=0.05,update_MoI_lr_decay=0.99,record_swaps = False,MoI=None,MoI_h=None,interpolate=False,degree_interpolate=None,reset_MoI=False,mavar_gamma=0.95):
        """
        Generate Monte Carlo samples from the RBM. Starting from random initial conditions, Gibbs updates are performed to sample from equilibrium.
        Inputs :
            Nchains (10): Number of Markov chains
            Lchains (100): Length of each chain
            Nthermalize (0): Number of Gibbs sampling steps to perform before the first sample of a chain.
            Nstep (1): Number of Gibbs sampling steps between each sample of a chain
            N_PT (1): Number of Monte Carlo Exchange replicas to use. This==useful if the mixing rate==slow. Watch self.acceptance_rates_g to check that it==useful (acceptance rates about 0==useless)
            batches (10): Number of batches. Must divide Nchains. higher==better for speed (more vectorized) but more RAM consuming.
            reshape (True): If True, the output==(Nchains x Lchains, n_visibles/ n_hiddens) (chains mixed). Else, the output==(Nchains, Lchains, n_visibles/ n_hiddens)
            config_init ([]). If not [], a Nchains X n_visibles numpy array that specifies initial conditions for the Markov chains.
            beta (1): The inverse temperature of the model.
        """
        if batches==None:
            batches = Nchains
        n_iter = Nchains//batches
        Ndata = Lchains * batches
        if record_replica:
            reshape = False

        if (N_PT>1):
            if record_acceptance==None:
                record_acceptance = True


            if update_betas ==None:
                update_betas = False

            if record_acceptance:
                self.mavar_gamma = mavar_gamma

            if update_betas:
                record_acceptance = True
                self.update_betas_lr = update_betas_lr
                self.update_betas_lr_decay = update_betas_lr_decay
            if (MoI is not None) | (MoI_h is not None):
                if MoI is not None:
                    self.from_MoI = True
                    self.from_MoI_h = False
                else:
                    self.from_MoI = False
                    self.from_MoI_h = True
                if interpolate:
                    self.update_interpolation_MoI_lr = update_MoI_lr
                    self.update_interpolation_MoI_lr_decay = update_MoI_lr_decay
                    if degree_interpolate is None:
                        degree_interpolate = N_PT-1
                if reset_MoI or not hasattr(self,'zlayer'):
                    if self.from_MoI:
                        self.init_zlayer_portal(MoI,interpolate=interpolate,degree_interpolate=degree_interpolate,layer_id=0)
                    else:
                        self.init_zlayer_portal(MoI_h,interpolate=interpolate,degree_interpolate=degree_interpolate,layer_id=1)

                self.n_layers +=1 # Add extra layer.
                self.layers_name.append('zlayer')
                self.layers_size.append(1)
                self.layers_nature.append('Potts')
                self.layers_n_c.append(self.zlayer.n_c)

            else:
                self.from_MoI= False
                self.from_MoI_h = False

        else:
            record_acceptance = False
            update_betas = False
            self.from_MoI = False
            self.from_MoI_h = False





        if (N_PT > 1) & record_replica:
            data = [ np.zeros([Nchains,N_PT,Lchains,self.layers_size[i]],dtype= getattr(self,self.layers_name[i]).type ) for i in range(self.n_layers) ]
        else:
            data = [ np.zeros([Nchains,Lchains,self.layers_size[i]],dtype= getattr(self,self.layers_name[i]).type ) for i in range(self.n_layers) ]

        if self.n_layers ==1:
            data = data[0]

        if config_init != []:
            if type(config_init) == np.ndarray:
                config_init_ = []
                for k,layer in enumerate(self.layers_name):
                    if config_init.shape[1] == self.layers_size[k]:
                        config_init_.append(config_init.copy())
                    else:
                        config_init_.append(getattr(self,layer).random_init_config(batches) )
                config_init = config_init_


        for i in range(n_iter):
            if config_init == []:
                config = self._gen_data(Nthermalize, Ndata,Nstep, N_PT = N_PT, batches = batches, reshape = False,beta = beta,record_replica = record_replica, record_acceptance=record_acceptance,update_betas=update_betas,record_swaps=record_swaps,interpolate=interpolate)
            else:
                config = self._gen_data(Nthermalize, Ndata,Nstep, N_PT = N_PT, batches = batches, reshape = False,beta = beta,record_replica = record_replica, config_init = [config_init[l][batches*i:batches*(i+1)] for l in range(self.n_layers)],record_acceptance=record_acceptance,update_betas=update_betas,record_swaps=record_swaps,interpolate=interpolate)
            if (N_PT > 1) & record_replica:
                if self.n_layers==1:
                    data[batches*i:batches*(i+1),:,:,:] = copy.copy(np.swapaxes(config,0,2))
                else:
                    for l in range(self.n_layers):
                        data[l][batches*i:batches*(i+1),:,:,:] = copy.copy(np.swapaxes(config[l],0,2))
            else:
                if self.n_layers==1:
                    data[batches*i:batches*(i+1),:,:] = copy.copy(np.swapaxes(config,0,1))
                else:
                    for l in range(self.n_layers):
                        data[l][batches*i:batches*(i+1),:,:] = copy.copy(np.swapaxes(config[l],0,1))


        if reshape:
            if self.n_layers==1:
                data =  data.reshape([Nchains*Lchains,self.layers_size[0]])
            else:
                data =  [ data[layer].reshape([Nchains*Lchains,self.layers_size[layer]]) for layer in range(self.n_layers)  ]

        if self.from_MoI | self.from_MoI_h: # Remove extra layer.
            self.n_layers -=1
            self.layers_name = self.layers_name[:-1]
            self.layers_size = self.layers_size[:-1]
            self.layers_nature = self.layers_nature[:-1]
            self.layers_n_c = self.layers_n_c[:-1]
            self.from_MoI = False
        return data

    def _gen_data(self,Nthermalize,Ndata,Nstep, N_PT =1, batches = 1,reshape = True,config_init=[],beta = 1, record_replica = False,record_acceptance = True,update_betas = False,record_swaps = False,interpolate=False):
        self.N_PT = N_PT
        if self.N_PT > 1:
            if update_betas | (not hasattr(self,'betas')):
                self.betas =  np.arange(N_PT)/(N_PT-1) * beta
                self.betas = self.betas[::-1].astype(curr_float)
            if (len(self.betas) != N_PT):
                self.betas =  np.arange(N_PT)/(N_PT-1) * beta
                self.betas = self.betas[::-1].astype(curr_float)

            self.acceptance_rates = np.zeros(N_PT-1,dtype=curr_float)
            self.mav_acceptance_rates = np.zeros(N_PT-1,dtype=curr_float)
        self.count_swaps = 0
        self.record_swaps = record_swaps
        if self.record_swaps:
            self.particle_id = [np.arange(N_PT)[:,np.newaxis].repeat(batches,axis=1)]

        Ndata = Ndata//batches
        if N_PT >1:
            config = [ getattr(self,layer).random_init_config(batches,N_PT=N_PT) for layer in self.layers_name]
            if config_init != []:
                for l in range(self.n_layers):
                    config[l][0] = config_init[l]
            if not (self.from_MoI | self.from_MoI_h):
                energy = self.energy(config,remove_init=True)
        else:
            if config_init != []:
                config = config_init
            else:
                config = [ getattr(self,layer).random_init_config(batches) for layer in self.layers_name]

        if self.n_layers==1:
            config = config[0] #no list


        for _ in range(Nthermalize):
            if N_PT > 1:            
                if self.from_MoI:
                    config = self.exchange_step_APT(config, record_acceptance = record_acceptance)
                    config = self.markov_step_APT(config,beta=self.betas,recompute=False)
                elif self.from_MoI_h:
                    config = self.exchange_step_APTh(config, record_acceptance = record_acceptance)
                    config = self.markov_step_APTh(config,beta=self.betas,recompute=False)
                else:
                    config,energy = self.exchange_step_PT(config,energy,record_acceptance=record_acceptance)
                    config,energy = self.markov_step_and_energy(config,energy,beta=self.betas,compute_energy=False)
                if update_betas:
                    self.update_betas(beta=beta)
                    self.update_betas_lr *= self.update_betas_lr_decay
                if interpolate:
                    self.update_interpolation_MoI(config[self.n_layers-1],self.betas,self.update_interpolation_MoI_lr)
                    self.update_interpolation_MoI_lr *= self.update_interpolation_MoI_lr_decay
            else:
                config = self.markov_step(config, beta = beta)
        if self.n_layers==1:
            data = [ utilities.copy_config(config,N_PT=N_PT,record_replica=record_replica)]
        else:
            data = [ [utilities.copy_config(config[l],N_PT=N_PT,record_replica=record_replica)] for l in range(self.n_layers)]

        for _ in range(Ndata-1):
            for _ in range(Nstep):
                    if N_PT > 1:
                        if self.from_MoI:
                            config = self.exchange_step_APT(config, record_acceptance = record_acceptance)
                            config = self.markov_step_APT(config,beta=self.betas, recompute=False)
                        elif self.from_MoI_h:
                            config = self.exchange_step_APTh(config, record_acceptance = record_acceptance)
                            config = self.markov_step_APTh(config,beta=self.betas, recompute= False)
                        else:
                            config,energy = self.exchange_step_PT(config,energy,record_acceptance=record_acceptance)
                            config,energy = self.markov_step_and_energy(config,energy, beta = self.betas,compute_energy=False)

                        if update_betas:
                            self.update_betas(beta=beta)
                            self.update_betas_lr *= self.update_betas_lr_decay
                        if interpolate:
                            self.update_interpolation_MoI(config[self.n_layers-1],self.betas,self.update_interpolation_MoI_lr)
                            self.update_interpolation_MoI_lr *= self.update_interpolation_MoI_lr_decay
                    else:
                        config = self.markov_step(config, beta = beta)
            if self.n_layers ==1:
                data.append( utilities.copy_config(config,N_PT=N_PT,record_replica=record_replica)  )
            else:
                for l in range(self.n_layers):
                    data[l].append( utilities.copy_config(config[l],N_PT=N_PT,record_replica=record_replica)  )


        if self.record_swaps:
            print('cleaning particle trajectories')
            positions = np.array(self.particle_id)
            invert = np.zeros([batches,Ndata+Nthermalize,N_PT])
            for b in range(batches):
                for i in range(Ndata+Nthermalize):
                    for k in range(N_PT):
                        invert[b,i,k] = np.nonzero( positions[i,:,b]==k)[0]
            self.particle_id = invert
            self.last_at_zero = np.zeros([batches,Ndata+Nthermalize,N_PT])
            for b in range(batches):
                for i in range(Ndata+Nthermalize):
                    for k in range(N_PT):
                        tmp = np.nonzero(self.particle_id[b,:i,k]==0)[0]
                        if len(tmp)>0:
                            self.last_at_zero[b,i,k] = i-1-tmp.max()
                        else:
                            self.last_at_zero[b,i,k] = i
            self.last_at_zero[:,0,0] = 0


            self.trip_duration = np.zeros([batches,Ndata+Nthermalize])
            for b in range(batches):
                for i in range(Ndata+Nthermalize):
                    self.trip_duration[b,i] = self.last_at_zero[b,i,np.nonzero(invert[b,i,:] == N_PT-1)[0]]




        if reshape:
            if self.n_layers==1:
                data= np.array(data).reshape([Ndata*batches,self.layers_size[0]])
            else:
                for l in range(self.n_layers):
                    data[l] = np.array(data[l]).reshape([Ndata*batches,self.layers_size[l]])
        else:
            if self.n_layers==1:
                data = np.array(data)
            else:
                for l in range(self.n_layers):
                    data[l] = np.array(data[l])
        return data



    def update_betas(self,beta=1):
        if self.N_PT <3:
            return
        else:
            if self.acceptance_rates.mean()>0:
                self.stiffness = np.maximum(1 - (self.mav_acceptance_rates/self.mav_acceptance_rates.mean()),0) + 1e-4 * np.ones(self.N_PT-1)
                diag = self.stiffness[0:-1] + self.stiffness[1:]
                if self.N_PT>3:
                    offdiag_g = -self.stiffness[1:-1]
                    offdiag_d = -self.stiffness[1:-1]
                    M = diags([offdiag_g,diag,offdiag_d],offsets = [-1,0,1],shape = [self.N_PT -2, self.N_PT-2]).toarray()
                else:
                    M = diags([diag],offsets=[0],shape = [self.N_PT -2, self.N_PT-2]).toarray()
                B = np.zeros(self.N_PT-2,dtype=curr_float)
                B[0] = self.stiffness[0] * beta
                self.betas[1:-1] = self.betas[1:-1] * (1- self.update_betas_lr) + self.update_betas_lr *  np.linalg.solve(M,B)
                # self.update_betas_lr*= self.update_betas_lr_decay

    def AIS(self,M = 10, n_betas = 10000, batches = None, verbose = 0, beta_type = 'adaptive',reset_betas=True):
        if beta_type == 'linear':
            betas = np.arange(n_betas)/float(n_betas-1)
        elif beta_type == 'root':
            betas = np.sqrt( np.arange(n_betas)/float(n_betas-1) )
        elif beta_type == 'adaptive':
            if not hasattr(self,'N_PT'):
                self.N_PT = 1
            copy_N_PT = copy.copy(self.N_PT)
            if hasattr(self,'betas'):
                if reset_betas:
                    tmp = 0
                else:
                    if self.N_PT%2:
                        tmp = 1
                    else:
                        tmp = 0
            else:
                tmp = 0

            if tmp:
                if verbose:
                    print('Using previously computed betas: %s'%self.betas)
                N_PT = len(self.betas)
                tmp2 = 0
            else:
                if hasattr(self,'betas'):
                    tmp2 = 1
                    copy_beta = self.betas.copy()
                else:
                    tmp2 = 0

                Nthermalize = 200
                Nchains = 20
                N_PT = 11
                self.adaptive_PT_lr = 0.05
                self.adaptive_PT_decay = True
                self.adaptive_PT_lr_decay = 10**(-1/float(Nthermalize))
                if verbose:
                    t = time.time()
                    print('Learning betas...')
                self.gen_data(N_PT = N_PT, Nchains = Nchains, Lchains =1, Nthermalize = Nthermalize,update_betas = True)
                if verbose:
                    print('Elapsed time: %s, Acceptance rates: %s'%(time.time() -t, self.mav_acceptance_rates))
            betas = []
            sparse_betas = self.betas[::-1]
            for i in range(N_PT - 1):
                betas+= list( sparse_betas[i] + (sparse_betas[i+1] - sparse_betas[i]) * np.arange(n_betas/(N_PT - 1))/float(n_betas/(N_PT - 1)-1) )
            betas = np.array(betas)
            n_betas = len(betas)
            # if verbose:
                # import matplotlib.pyplot as plt
                # plt.plot(betas); plt.title('Interpolating temperatures');plt.show()

            # Initialization.
        log_weights =  np.zeros(M,dtype=curr_float)
        config = []
        layers_name = self.layers_name
        layers_size = self.layers_size
        layers_n_c = self.layers_n_c
        layers_nature = self.layers_nature
        n_layers = self.n_layers


        for layer,N,n_c in zip(layers_name,layers_size,layers_n_c):
            if n_c>1:
                config.append(getattr(self,layer).sample_from_inputs(np.zeros([M,N,n_c],dtype=curr_float),beta=0) )
            else:
                config.append(getattr(self,layer).sample_from_inputs(np.zeros([M,N],dtype=curr_float),beta=0) )

        if n_layers ==1:
            fields_eff = self.compute_fields_eff(config[0][0])
            config = (config[0][0],fields_eff)
        energy = np.zeros(M,dtype=curr_float)

        log_Z_init = np.zeros(1,dtype=curr_float)
        for N,layer in zip(layers_size, layers_name):
            log_Z_init += getattr(self,layer).logpartition(None, beta=0)

        if verbose:
            print('Initial evaluation: log(Z) = %s'%log_Z_init)

        for i in range(1,n_betas):
            if verbose:
                if (i%2000==0):
                    print('Iteration %s, beta: %s'%(i,betas[i]) )
                    print('Current evaluation: log(Z)= %s +- %s'%( (log_Z_init + log_weights).mean(), (log_Z_init + log_weights).std()/np.sqrt(M)) )

            config,energy = self.markov_step_and_energy(config,energy,beta=betas[i])
            log_weights += -(betas[i] - betas[i-1]) * energy
        self.log_Z_AIS = (log_Z_init + log_weights).mean()
        self.log_Z_AIS_std = (log_Z_init + log_weights).std()/np.sqrt(M)

        if beta_type == 'adaptive':
            self.N_PT = copy_N_PT
            if tmp2:
                self.betas = copy_beta

        if verbose:
            print('Final evaluation: log(Z)= %s +- %s'%(self.log_Z_AIS, self.log_Z_AIS_std) )
        return self.log_Z_AIS, self.log_Z_AIS_std

    def likelihood(self,data,recompute_Z = False):
        if (not hasattr(self,'log_Z_AIS')) | recompute_Z:
            self.AIS()
        return -self.free_energy(data) - self.log_Z_AIS


    def init_zlayer_portal(self,MoI,interpolate=False,degree_interpolate=2,from_start=False,layer_id=0):
        xlayer = getattr(self,self.layers_name[layer_id])
        nature = xlayer.nature
        n_c = xlayer.n_c

        if interpolate:
            zlayer = layer.initLayer(N=1, nature = 'PottsInterpolate', n_c = MoI.M,degree=degree_interpolate)
        else:
            zlayer = layer.initLayer(N=1, nature = 'Potts', n_c = MoI.M)

        if nature in ['Bernoulli','Spin','Potts']:
            tmp = MoI.weights -  xlayer.fields0[np.newaxis]

        if nature == 'Potts':
            weights_MoI = np.asarray(np.swapaxes(tmp[np.newaxis,:,:,:],1,2 ), order='c' )
        else:
            weights_MoI = np.asarray(np.swapaxes(tmp[np.newaxis,:,:],1,2 ), order='c' )


        fields = MoI.gh.copy()[np.newaxis,:].astype(curr_float)
        fields0 = MoI.gh - xlayer.logpartition(None,I0=tmp,beta=0)
        fields0 -= fields0.mean()
        fields0 = fields0[np.newaxis,:].astype(curr_float)

        if interpolate:
            zlayer.fields[0] = fields0
            zlayer.fields[1] = fields - fields0
            zlayer.mu_ref = MoI.muh[np.newaxis,:].astype(curr_float)
        else:
            zlayer.fields = fields
            zlayer.fields0 = fields0

        self.zlayer = zlayer
        self.weights_MoI = weights_MoI.astype(curr_float)
        self.n_z = MoI.M
        self.muz = np.asarray(MoI.muh[np.newaxis,:],order='c',dtype=curr_float)
        if nature == 'Potts':
            self.muxz = np.asarray( np.swapaxes(MoI.cond_muv,0,1)[np.newaxis,:,:,:] * self.muz[:,np.newaxis,:,np.newaxis] , dtype=curr_float,order='c')
        else:
            self.muxz = np.asarray( np.swapaxes(MoI.cond_muv,0,1)[np.newaxis,:,:] * self.muz[:,np.newaxis,:] , dtype=curr_float, order='c')

        if nature in ['Gaussian','ReLU','ReLU+','dReLU']:
            self.mux2 = MoI.muv2.copy().astype(curr_float)
        if nature in ['ReLU','ReLU+']:
            self.muxabs = MoI.muvabs.copy().astype(curr_float)
        if nature == 'dReLU':
            self.mux2pm = MoI.mux2pm.copy().astype(curr_float)



    def update_moments_MoI(self,x,layer_id=0,eps= 1e-5):
        xlayer = getattr(self,self.layers_name[layer_id])
        nature = xlayer.nature
        z = self.zlayer.mean_from_inputs(np.zeros([self.nchains,1,self.zlayer.n_c],dtype=curr_float),I0=xlayer.compute_output(x,self.weights_MoI),beta=0)
        self.muz = self.update_MoI_lr * z.mean(0) + (1-self.update_MoI_lr) * self.muz
        self.muxz = self.update_MoI_lr * utilities.average_product(z,x,mean1=True,c1=self.zlayer.n_c,mean2=False,c2=xlayer.n_c)  + (1-self.update_MoI_lr) * self.muxz
        if nature in ['Gaussian','ReLU','ReLU+','dReLU']:
            self.mux2 = self.update_MoI_lr * (x**2).mean(0) + (1-self.update_MoI_lr) * self.mux2
        if nature in ['ReLU','ReLU+']:
            self.muxabs = self.update_MoI_lr * np.abs(x).mean(0) + (1-self.update_MoI_lr) * self.muxabs
        if nature == 'dReLU':
            self.mux2pm = self.update_MoI_lr * (np.maximum(x,0)**2 - np.minimum(x,0)**2 ) + (1-self.update_MoI_lr) * self.mux2pm


        if nature == 'Potts':
            self.cond_mux = self.muxz/(self.muz[:,np.newaxis,:,np.newaxis]+eps)
        else:
            self.cond_mux = self.muxz/(self.muz[:,np.newaxis,:]+eps)









    def update_params_MoI(self,layer_id =0 , eps = 1e-5):
        xlayer = getattr(self,self.layers_name[layer_id])
        nature = xlayer.nature

        k = np.argmin(self.muz)
        if self.muz[0,k] < 0.1/self.zlayer.n_c:
            print('Reloading mixture %s, %.2f'%(k,self.muz[0,k]) )
            k_max = np.argmax(self.muz)
            self.muz[0,k_max] = (self.muz[0,k_max] + self.muz[0,k])/2
            self.muz[0,k] = self.muz[0,k_max].copy()
            muxz_ref = self.muxz[:,:,k_max] + self.muxz[:,:,k]
            if nature == 'Potts':
                noise = np.random.rand(1,xlayer.N,xlayer.n_c)
                noise/= noise.sum(-1)[:,:,np.newaxis]
            elif nature == 'Bernoulli':
                noise = np.random.rand(1,xlayer.N)
                noise/= noise.sum(-1)
            elif nature == 'Spin':
                noise = (2*np.random.rand(1,xlayer.N) -1)

            self.muxz[:,:,k_max] = muxz_ref/2
            self.muxz[:,:,k] = 0.95 * muxz_ref/2 + 0.05 * noise * self.muz[0,k]
            if nature == 'Potts':
                self.cond_mux[:,:,k_max] = self.muxz[:,:,k_max]/(self.muz[:,np.newaxis,k_max,np.newaxis]+eps)
                self.cond_mux[:,:,k] = self.muxz[:,:,k]/(self.muz[:,np.newaxis,k,np.newaxis]+eps)
            else:
                self.cond_mux[:,:,k_max] = self.muxz[:,:,k_max]/(self.muz[:,np.newaxis,k_max]+eps)
                self.cond_mux[:,:,k] = self.muxz[:,:,k]/(self.muz[:,np.newaxis,k]+eps)

            if self.interpolate:
                self.zlayer.fields[2:,:,k] = self.zlayer.fields[2:,:,k_max]
                self.zlayer.fields[2:] -= self.zlayer.fields[2:].mean()[np.newaxis]



        if nature == 'Bernoulli':
            self.weights_MoI = np.log((self.cond_mux+eps)/(1-self.cond_mux + eps)) - xlayer.fields0[np.newaxis,:,np.newaxis]
        elif nature == 'Spin':
            self.weights_MoI = 0.5 * np.log((1+self.cond_mux+eps)/(1-self.cond_mux + eps))  - xlayer.fields0[np.newaxis,:,np.newaxis]
        elif nature == 'Potts':
            self.weights_MoI = np.log(self.cond_mux + eps) - xlayer.fields0[np.newaxis,:,np.newaxis,:]
            self.weights_MoI -= self.weights_MoI.mean(-1)[:,:,:,np.newaxis]

        self.weights_MoI = np.asarray(self.weights_MoI,order='c')

        self.zlayer.mu_ref = self.muz
        gz = np.log(self.muz+eps)
        gz -= gz.mean()
        gz0 = gz - xlayer.logpartition( None , psi0 = np.swapaxes(self.weights_MoI[0],0,1 ) , beta=0 )
        gz0 -= gz0.mean()
        if self.interpolate:
            self.zlayer.fields[0] = gz0
            self.zlayer.fields[1] = gz - gz0
        else:
            self.zlayer.fields = gz
            self.zlayer.fields0 = gz0


    def update_interpolation_MoI(self,zs,betas,learning_rate):
        if (self.zlayer.degree<2) | len(betas)<3:
            return
        else:
            nbetas = len(betas)
            mu = np.array([utilities.average(zs[l],c=self.zlayer.n_c) for l in range(1,nbetas-1)])
            var = mu * (1-mu) + 1.0/zs.shape[1]
            coefficients = self.zlayer.get_coefficients(betas)[2:,1:-1]
            P1 = np.linalg.pinv(coefficients,rcond=0.1)
            P2 = np.linalg.pinv(coefficients.T,rcond=0.1)
            grad = coefficients.sum(1)[:,np.newaxis,np.newaxis] * self.zlayer.mu_ref[np.newaxis,:,:] -np.tensordot(coefficients, mu,axes=(1,0) )
            direction = np.tensordot(P2,np.tensordot(P1,grad,axes=(1,0))/var, axes=(1,0) )
            direction -= direction.mean(-1)[:,:,np.newaxis]
            self.zlayer.fields[2:] += learning_rate * direction
