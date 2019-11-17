"""
Created on Thu Oct 27 11:46:59 2016

RBM class.
@author: jerometubiana
"""


import numpy as np
import layer7 as layer
import pgm7 as pgm
import moi
import utilities7 as utilities
from utilities7 import check_random_state,gen_even_slices,logsumexp,log_logistic,average,average_product,saturate
import time,copy
import itertools

double_precision = False

if double_precision:
    curr_float = np.float64
    curr_int = np.int64
else:
    curr_float = np.float32
    curr_int = np.int16

#%%

class RBM(pgm.PGM):
    def __init__(self,n_v = 100, n_h = 20,visible = 'Bernoulli', hidden='Bernoulli', n_cv =1, n_ch =1, random_state = None, gauge='zerosum',zero_field = False):
        self.n_v = n_v
        self.n_h = n_h
        self.n_visibles = n_v
        self.n_hiddens = n_h
        self.visible = visible
        self.hidden =hidden
        self.random_state = check_random_state(random_state)
        if self.visible == 'Potts':
            self.n_cv = n_cv
        else:
            self.n_cv = 1
        if self.hidden == 'Potts':
            self.n_ch = n_ch
        else:
            self.n_ch = 1

        super(RBM, self).__init__(n_layers = 2, layers_size = [self.n_v,self.n_h],layers_nature = [visible,hidden], layers_n_c = [self.n_cv,self.n_ch] , layers_name = ['vlayer','hlayer'] )


        self.gauge = gauge
        self.zero_field = zero_field
        self.vlayer = layer.initLayer(N= self.n_v, nature = self.visible, position = 'visible', n_c = self.n_cv, random_state = self.random_state, zero_field = self.zero_field)
        self.hlayer = layer.initLayer(N= self.n_h, nature = self.hidden, position = 'hidden', n_c = self.n_ch, random_state = self.random_state, zero_field = self.zero_field)
        self.init_weights(0.01)
        self.tmp_l2_fields = 0
        self._Ivh = None
        self._Ihv = None
        self._Ivz = None
        self._Izv = None
        self._Ihz = None
        self._Izh = None



    def init_weights(self,amplitude):
        if (self.n_ch >1) & (self.n_cv>1):
            self.weights = amplitude * self.random_state.randn(self.n_h, self.n_v,self.n_ch, self.n_cv)
            self.weights = pgm.gauge_adjust_couplings(self.weights,self.n_ch,self.n_cv,gauge=self.gauge)
        elif (self.n_ch >1) & (self.n_cv ==1):
            self.weights = amplitude * self.random_state.randn(self.n_h, self.n_v,self.n_ch)
            self.weights = pgm.gauge_adjust_couplings(self.weights,self.n_ch,self.n_cv,gauge=self.gauge)
        elif (self.n_ch ==1) & (self.n_cv>1):
            self.weights = amplitude * self.random_state.randn(self.n_h, self.n_v,self.n_cv)
            self.weights = pgm.gauge_adjust_couplings(self.weights,self.n_ch,self.n_cv,gauge=self.gauge)
        else:
            self.weights = amplitude * self.random_state.randn(self.n_h, self.n_v)
        self.weights = np.asarray(self.weights,dtype=curr_float)


    # def markov_step(self,x,beta =1):
    #     (v,h) = x
    #     h = self.hlayer.sample_from_inputs( self.vlayer.compute_output(v,self.weights, direction ='up') , beta = beta )
    #     v = self.vlayer.sample_from_inputs( self.hlayer.compute_output(h,self.weights,direction='down') , beta = beta )
    #     return (v,h)

    def markov_step(self,x,beta =1):
        (v,h) = x
        self._Ivh = self.vlayer.compute_output(v,self.weights, direction ='up',out=self._Ivh)
        h = self.hlayer.sample_from_inputs( self._Ivh , beta = beta, out = h )
        self._Ihv = self.hlayer.compute_output(h,self.weights,direction='down',out=self._Ihv)
        v = self.vlayer.sample_from_inputs( self._Ihv , beta = beta, out = v )
        return (v,h)

    def markov_step_and_energy(self, x,E, beta=1):
        (v,h) = x
        h = self.hlayer.sample_from_inputs( self.vlayer.compute_output(v,self.weights,direction='up'), beta=beta )
        I = self.hlayer.compute_output(h,self.weights,direction='down')
        v,E = self.vlayer.sample_and_energy_from_inputs(I,beta=beta,remove_init=True)
        E+= self.hlayer.energy(h,remove_init=True)
        return (v,h),E

    def markov_step_APT(self, x,  beta=1,recompute=True):
        (v,h,z) = x
        beta_is_array = (type(beta)==np.ndarray)
        if beta_is_array:
            N_PT = v.shape[0]
            B = v.shape[1]
        else:
            N_PT = 1
            B = v.shape[0]
        if recompute:
            self._Ivz = self.vlayer.compute_output(v,self.weights_MoI,direction='up',out=self._Ivz)
            self._Ivh = self.vlayer.compute_output(v,self.weights,direction='up',out=self._Ivh)
        h = self.hlayer.sample_from_inputs(self._Ivh,beta=beta)
        if beta_is_array: # many temperatures. Last one at beta=0 = MoI configuration, obtained by direct sampling.
            z[-1] = self.zlayer.sample_from_inputs(np.zeros([B,1,self.n_z],dtype=curr_float),beta=1)
            z[:-1] = self.zlayer.sample_from_inputs(np.zeros([N_PT-1,B,1,self.n_z],dtype=curr_float),I0=self._Ivz[:-1],beta=beta[:-1])
        else:
            if beta==0:
                z = self.zlayer.sample_from_inputs(np.zeros([B,1,self.n_z],dtype=curr_float),beta=1)
            else:
                z = self.zlayer.sample_from_inputs(np.zeros([B,1,self.n_z],I0=self._Ivz,dtype=curr_float),beta=beta)

        self._Izv = self.zlayer.compute_output(z, self.weights_MoI,direction='down',out = self._Izv)
        self._Ihv =self.hlayer.compute_output(h,self.weights,direction='down',out= self._Ihv)
        v = self.vlayer.sample_from_inputs(self._Ihv, I0=self._Izv ,beta = beta)
        return (v,h,z)

    def markov_step_APTh(self, x,  beta=1,recompute=True):
        (v,h,z) = x
        beta_is_array = (type(beta)==np.ndarray)
        if beta_is_array:
            N_PT = v.shape[0]
            B = v.shape[1]
        else:
            N_PT = 1
            B = v.shape[0]
        if recompute:
            self._Ihz = self.hlayer.compute_output(h,self.weights_MoI,direction='up',out=self._Ihz)
            self._Ihv = self.hlayer.compute_output(h,self.weights,direction='down',out=self._Ihz)
        v = self.vlayer.sample_from_inputs(self._Ihv, beta = beta)
        if beta_is_array:
            z[-1] = self.zlayer.sample_from_inputs(np.zeros([B,1,self.n_z],dtype=curr_float),beta=1)
            z[:-1] = self.zlayer.sample_from_inputs(np.zeros([N_PT-1,B,1,self.n_z],dtype=curr_float),I0=self._Ihz[:-1],beta=beta[:-1])
        else:
            if beta==0:
                z = self.zlayer.sample_from_inputs(np.zeros([B,1,self.n_z],dtype=curr_float),beta=1)
            else:
                z = self.zlayer.sample_from_inputs(np.zeros([B,1,self.n_z],I0=self._Ihz,dtype=curr_float),beta=beta)

        self._Ivh = self.vlayer.compute_output(v,self.weights,direction='up',out=self._Ivh)
        self._Izh = self.zlayer.compute_output(z, self.weights_MoI,direction='down',out=self._Izh)
        h = self.hlayer.sample_from_inputs(self._Ivh,I0=self._Izh,beta=beta)
        return (v,h,z)

    def exchange_step_PT(self,x,E,record_acceptance=True,compute_energy=True):
        (v,h) = x
        if compute_energy:
            E = self.energy( (v,h),remove_init=True)
        if self.record_swaps:
            particle_id = self.particle_id[-1].copy()
        for i in np.arange(self.count_swaps%2,self.N_PT-1,2):
            proba = np.minimum( 1,  np.exp( (self.betas[i+1]-self.betas[i]) * (E[i+1,:]-E[i,:])   ) )
            swap = self.random_state.rand(proba.shape[0]) < proba
            if i>0:
                v[i:i+2,swap,:] = v[i+1:i-1:-1,swap ,:]
                h[i:i+2,swap,:] = h[i+1:i-1:-1,swap,:]
                E[i:i+2,swap] = E[i+1:i-1:-1,swap]
                if self.record_swaps:
                    particle_id[i:i+2,swap] = particle_id[i+1:i-1:-1,swap]

            else:
                v[i:i+2,swap,:] = v[i+1::-1,swap,:]
                h[i:i+2,swap,:] = h[i+1::-1,swap,:]
                E[i:i+2,swap] = E[i+1::-1,swap]
                if self.record_swaps:
                    particle_id[i:i+2,swap] = particle_id[i+1::-1,swap]


            if record_acceptance:
                self.acceptance_rates[i] = swap.mean()
                self.mav_acceptance_rates[i] = self.mavar_gamma * self.mav_acceptance_rates[i] +  self.acceptance_rates[i]*(1-self.mavar_gamma)

        if self.record_swaps:
            self.particle_id.append(particle_id)

        self.count_swaps +=1
        return (v,h),E




    def exchange_step_APT(self,x, record_acceptance = True):
        (v,h,z) = x
        self._Ivh = self.vlayer.compute_output(v,self.weights,direction='up',out=self._Ivh)
        self._Ivz = self.vlayer.compute_output(v,self.weights_MoI,direction='up',out=self._Ivz)

        if self.record_swaps:
            particle_id = self.particle_id[-1].copy()


        for i in range(self.count_swaps%2,self.N_PT-1,2):
            F_11 = self.free_energy_APT(v[i], Ivz = self._Ivz[i],Ivh=self._Ivh[i], beta=self.betas[i])
            F_12 = self.free_energy_APT(v[i+1], Ivz = self._Ivz[i+1],Ivh=self._Ivh[i+1], beta=self.betas[i])
            F_22 = self.free_energy_APT(v[i+1], Ivz = self._Ivz[i+1],Ivh=self._Ivh[i+1], beta=self.betas[i+1])
            F_21 = self.free_energy_APT(v[i], Ivz = self._Ivz[i],Ivh=self._Ivh[i], beta=self.betas[i+1])
            proba = np.minimum(1, np.exp( - (F_21 + F_12 - F_11 - F_22) ) )
            swap = self.random_state.rand(proba.shape[0]) < proba
            if i>0:
                v[i:i+2,swap] = v[i+1:i-1:-1,swap]
                h[i:i+2,swap] = h[i+1:i-1:-1,swap]
                z[i:i+2,swap] = z[i+1:i-1:-1,swap]
                self._Ivh[i:i+2,swap] = self._Ivh[i+1:i-1:-1,swap]
                self._Ivz[i:i+2,swap] = self._Ivz[i+1:i-1:-1,swap]
                if self.record_swaps:
                    particle_id[i:i+2,swap] = particle_id[i+1:i-1:-1,swap]
            else:
                v[i:i+2,swap] = v[i+1::-1,swap]
                h[i:i+2,swap] = h[i+1::-1,swap]
                z[i:i+2,swap] = z[i+1::-1,swap]
                self._Ivh[i:i+2,swap] = self._Ivh[i+1::-1,swap]
                self._Ivz[i:i+2,swap] = self._Ivz[i+1::-1,swap]
                if self.record_swaps:
                    particle_id[i:i+2,swap] = particle_id[i+1::-1,swap]


            if record_acceptance:
                self.acceptance_rates[i] = swap.mean()
                self.mav_acceptance_rates[i] = self.mavar_gamma * self.mav_acceptance_rates[i] +  self.acceptance_rates[i]*(1-self.mavar_gamma)


        if self.record_swaps:
            self.particle_id.append(particle_id)

        self.count_swaps +=1
        return (v,h,z)



    def exchange_step_APTh(self,x, record_acceptance = True):
        (v,h,z) = x
        self._Ihv = self.hlayer.compute_output(h,self.weights,direction='down',out=self._Ihv)
        self._Ihz = self.hlayer.compute_output(h,self.weights_MoI,direction='up',out=self._Ihz)

        if self.record_swaps:
            particle_id = self.particle_id[-1].copy()


        for i in range(self.count_swaps%2,self.N_PT-1,2):
            F_11 = self.free_energy_APTh(h[i], Ihv = self._Ihv[i],Ihz=self._Ihz[i], beta=self.betas[i])
            F_12 = self.free_energy_APTh(h[i+1], Ihv = self._Ihv[i+1],Ihz=self._Ihz[i+1], beta=self.betas[i])
            F_22 = self.free_energy_APTh(h[i+1], Ihv = self._Ihv[i+1],Ihz=self._Ihz[i+1], beta=self.betas[i+1])
            F_21 = self.free_energy_APTh(h[i], Ihv = self._Ihv[i],Ihz=self._Ihz[i], beta=self.betas[i+1])
            proba = np.minimum(1, np.exp( - (F_21 + F_12 - F_11 - F_22) ) )
            swap = self.random_state.rand(proba.shape[0]) < proba
            if i>0:
                v[i:i+2,swap] = v[i+1:i-1:-1,swap]
                h[i:i+2,swap] = h[i+1:i-1:-1,swap]
                z[i:i+2,swap] = z[i+1:i-1:-1,swap]
                Ihv[i:i+2,swap] = Ihv[i+1:i-1:-1,swap]
                Ihz[i:i+2,swap] = Ihz[i+1:i-1:-1,swap]
                if self.record_swaps:
                    particle_id[i:i+2,swap] = particle_id[i+1:i-1:-1,swap]

            else:
                v[i:i+2,swap] = v[i+1::-1,swap]
                h[i:i+2,swap] = h[i+1::-1,swap]
                z[i:i+2,swap] = z[i+1::-1,swap]
                Ihv[i:i+2,swap] = Ihv[i+1::-1,swap]
                Ihz[i:i+2,swap] = Ihz[i+1::-1,swap]
                if self.record_swaps:
                    particle_id[i:i+2,swap] = particle_id[i+1::-1,swap]


            if record_acceptance:
                self.acceptance_rates[i] = swap.mean()
                self.mav_acceptance_rates[i] = self.mavar_gamma * self.mav_acceptance_rates[i] +  self.acceptance_rates[i]*(1-self.mavar_gamma)


        if self.record_swaps:
            self.particle_id.append(particle_id)

        self.count_swaps +=1
        return (v,h,z)


    def input_hiddens(self,v):
        if v.ndim ==1: v = v[np.newaxis,:]
        return self.vlayer.compute_output(v,self.weights, direction = 'up')


    def mean_hiddens(self,v):
        if v.ndim ==1: v = v[np.newaxis,:]
        return self.hlayer.mean_from_inputs( self.vlayer.compute_output(v,self.weights, direction = 'up'))

    def mean_visibles(self,h):
        if h.ndim ==1: h = h[np.newaxis,:]
        return self.vlayer.mean_from_inputs( self.hlayer.compute_output(h,self.weights, direction = 'down'))

    def sample_hiddens(self,v):
        if v.ndim ==1: v = v[np.newaxis,:]
        return self.hlayer.sample_from_inputs( self.vlayer.compute_output(v,self.weights, direction = 'up'))

    def sample_visibles(self,h):
        if h.ndim ==1: h = h[np.newaxis,:]
        return self.vlayer.sample_from_inputs( self.hlayer.compute_output(h,self.weights, direction = 'down'))

    def energy(self,x,remove_init = False):
        (v,h) = x
        if v.ndim ==1: v = v[np.newaxis,:]
        if h.ndim ==1: h = h[np.newaxis,:]
        return self.vlayer.energy(v,remove_init = remove_init) + self.hlayer.energy(h,remove_init = remove_init) - utilities.bilinear_form(self.weights,h,v,c1=self.n_ch,c2= self.n_cv)


    def free_energy(self,v,beta=1):
        if v.ndim ==1: v = v[np.newaxis,:]
        return self.vlayer.energy(v,beta=beta,remove_init=False) - self.hlayer.logpartition( self.vlayer.compute_output(v,self.weights,direction='up') ,beta=beta)


    def free_energy_h(self,h):
        if h.ndim ==1: h = h[np.newaxis,:]
        return self.hlayer.energy(h) - self.vlayer.logpartition(self.hlayer.compute_output(h,self.weights,direction='down'))

    def free_energy_APT(self,v, Ivh = None,Ivz=None, beta=1):
        if Ivh is None:
            Ivh = self.vlayer.compute_output(v,self.weights,direction='up')
        if Ivz is None:
            Ivz = self.vlayer.compute_output(v,self.weights_MoI,direction='up')


        F = self.vlayer.energy(v,beta=beta,remove_init=False)
        F -= self.zlayer.logpartition(np.zeros(Ivz.shape,dtype=curr_float),I0=Ivz,beta=beta)
        F -= self.hlayer.logpartition(Ivh,beta=beta)
        return F

    def free_energy_APTh(self,h, Ihv = None,Ihz=None, beta=1):
        if Ihv is None:
            Ihv = self.hlayer.compute_output(h,self.weights,direction='down')
        if Ihz is None:
            Ihz = self.hlayer.compute_output(h,self.weights_MoI,direction='up')

        F = self.hlayer.energy(h,beta=beta,remove_init=False)
        F -= self.zlayer.logpartition(np.zeros(Ihz.shape,dtype=curr_float),I0=Ihz,beta=beta)
        F -= self.vlayer.logpartition(Ihv,beta=beta)
        return F


    def compute_all_moments(self,from_hidden = True): # Compute all moments for RBMs with small number of hidden units.
        if self.hidden in ['ReLU','Gaussian' ,'ReLU+','dReLU']:
            from_hidden = False

        if from_hidden:
            configurations = utilities.make_all_discrete_configs(self.n_h,self.hidden,c=self.n_ch)
            weights = -self.free_energy_h(configurations)
            maxi = weights.max()
            weights -= maxi
            weights = np.exp(weights)
            logZ = np.log(weights.sum()) + maxi
            mean_hiddens = average(configurations,c = self.n_ch,weights = weights)
            mean_visibles = average(self.mean_visibles(configurations),weights = weights)
            covariance = average_product(configurations, self.mean_visibles(configurations),c1 = self.n_ch,c2=self.n_cv, mean1=False,mean2=True,weights=weights)
            return logZ,mean_visibles,mean_hiddens,covariance
        else:
            configurations = utilities.make_all_discrete_configs(self.n_v,self.visible,c=self.n_cv)
            weights = -self.free_energy(configurations)
            maxi = weights.max()
            weights -= maxi
            weights = np.exp(weights)
            logZ = np.log(weights.sum()) + maxi
            mean_visibles =  average(configurations,c = self.n_cv,weights = weights)
            mean_hiddens = average(self.mean_hiddens(configurations),weights = weights)
            covariance = average_product(self.mean_hiddens(configurations),configurations,c1 = self.n_ch,c2=self.n_cv, mean1=True,mean2=False,weights=weights)
            return logZ,mean_visibles,mean_hiddens,covariance





    def pseudo_likelihood(self,v):
        if self.visible not in ['Bernoulli','Spin','Potts','Bernoulli_coupled','Spin_coupled','Potts_coupled']:
            print('PL not supported for continuous data')
        else:
            if self.visible == 'Bernoulli':
                ind = (np.arange(v.shape[0]),self.random_state.randint(0, self.n_v, v.shape[0]))
                v_ = v.copy()
                v_[ind] = 1-v[ind]
                fe = self.free_energy(v)
                fe_ = self.free_energy(v_)
                return log_logistic(fe_ - fe)
            elif self.visible =='Spin':
                ind = (np.arange(v.shape[0]),self.random_state.randint(0, self.n_v, v.shape[0]))
                v_ = v.copy()
                v_[ind] = - v[ind]
                fe = self.free_energy(v)
                fe_ = self.free_energy(v_)
                return log_logistic(fe_ - fe)
            elif self.visible =='Potts':
                config = v
                ind_x = np.arange(config.shape[0])
                ind_y = self.random_state.randint(0, self.n_v, config.shape[0])
                E_vlayer_ref = self.vlayer.energy(config) + self.vlayer.fields[ind_y,config[ind_x,ind_y]]
                output_ref = self.vlayer.compute_output(config,self.weights) - self.weights[:,ind_y,config[ind_x,ind_y]].T
                fe = np.zeros([config.shape[0], self.n_cv],dtype=curr_float)
                for c in range(self.n_cv):
                    output = output_ref + self.weights[:,ind_y,c].T
                    E_vlayer = E_vlayer_ref - self.vlayer.fields[ind_y,c]
                    fe[:,c] = E_vlayer-self.hlayer.logpartition(output)
                return - fe[ind_x,config[ind_x,ind_y]] - logsumexp(- fe,1)






    def fit(self,data, batch_size = 100, learning_rate = None, extra_params = None, init='independent', optimizer='SGD', batch_norm=None,CD = False,N_PT = 1, N_MC = 1, nchains = None, n_iter = 10, MoI = 0,MoI_h=0,MoI_tau=None,interpolate=False,degree_interpolate=None, update_params_MoI = False,
            lr_decay = True,lr_final=None,decay_after = 0.5,l1 = 0, l1b = 0, l1c=0, l2 = 0,l2_fields =0,reg_theta=0,no_fields = False,weights = None, adapt_PT=False,AR_min = 0.3, adapt_MC = False, tau_max = 10, update_every = 100, N_PT_max = 20, N_MC_max = 20,from_hidden = False,
            update_betas =None, record_acceptance = None, shuffle_data = True,epsilon=  1e-6, verbose = 1, vverbose=0, record = [],record_interval = 100,data_test = None,weights_test=None,l1_custom=None,l1b_custom=None,M_AIS=10,n_betas_AIS=10000,decay_style='geometric'):

        self.batch_size = batch_size
        self.optimizer  = optimizer
        if self.hidden in ['Gaussian','ReLU','ReLU+']:
            if batch_norm is None:
                batch_norm = True
        else:
            if batch_norm is None:
                batch_norm = True
        self.batch_norm = batch_norm
        self.hlayer.batch_norm = batch_norm
        self.record_swaps = False

        self.n_iter = n_iter
        if self.n_iter == 1:
            lr_decay = False

        if learning_rate is None:
            if self.hidden in ['Bernoulli','Spin','Potts']:
                learning_rate = 0.1
            else:
                if self.batch_norm:
                    learning_rate = 0.1
                else:
                    learning_rate = 0.01

            if self.optimizer == 'ADAM':
                learning_rate *= 0.1

        self.learning_rate_init = copy.copy(learning_rate)
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        if self.lr_decay:
            self.decay_after = decay_after
            self.start_decay = int(self.n_iter*self.decay_after)
            if lr_final is None:
                self.lr_final = 1e-2 * self.learning_rate
            else:
                self.lr_final = lr_final
            self.decay_gamma = (float(self.lr_final)/float(self.learning_rate))**(1/float(self.n_iter* (1-self.decay_after) ))
        else:
            self.decay_gamma = 1

        self.no_fields = no_fields
        self.gradient = self.initialize_gradient_dictionary()
        self.do_grad_updates = {'vlayer':self.vlayer.do_grad_updates, 'weights':True}
        if self.batch_norm:
            self.do_grad_updates['hlayer'] = self.hlayer.do_grad_updates_batch_norm
        else:
            self.do_grad_updates['hlayer'] = self.hlayer.do_grad_updates


        if self.optimizer =='momentum':
            if extra_params is None:
                extra_params = 0.9
            self.momentum = extra_params
            self.previous_update = self.initialize_gradient_dictionary()



        elif self.optimizer == 'ADAM':
            if extra_params is None:
                extra_params = [0.9, 0.999, 1e-8]
            self.beta1 = extra_params[0]
            self.beta2 = extra_params[1]
            self.epsilon = extra_params[2]

            self.gradient_moment1 = self.initialize_gradient_dictionary()
            self.gradient_moment2 = self.initialize_gradient_dictionary()



        data = np.asarray(data,dtype=self.vlayer.type,order="c")
        if self.batch_norm:
            self.mu_data = utilities.average(data,c=self.n_cv,weights=weights)


        n_samples = data.shape[0]
        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        batch_slices = list(gen_even_slices(n_batches * self.batch_size,
                                            n_batches, n_samples))


        if init != 'previous':
            norm_init = np.sqrt(0.1/self.n_v)

            self.init_weights(norm_init)
            if init=='independent':
                self.vlayer.init_params_from_data(data,eps=epsilon,weights=weights)
            self.hlayer.init_params_from_data(None)


        self.adapt_PT = adapt_PT
        self.AR_min = AR_min
        self.N_PT_max = N_PT_max
        if self.adapt_PT:
            N_PT  = 2
            update_betas = True
            record_acceptance = True
            interpolate = True
        self.adapt_MC = adapt_MC
        self.tau_max = tau_max
        self.N_MC_max = N_MC_max
        if self.adapt_MC:
            N_MC=1



        self.N_PT = N_PT
        self.N_MC = N_MC
        if N_MC == 0:
            self.from_hidden = from_hidden
            if self.from_hidden:
                if self.n_ch >1:
                    nchains = (self.n_ch)**self.n_h
                else:
                    nchains = 2**self.n_h
            else:
                if self.n_cv > 1:
                    nchains = (self.n_cv)**self.n_v
                else:
                    nchains = 2**self.n_v
        else:
            self.from_hidden = False

        if nchains is None:
            self.nchains = self.batch_size
        else:
            self.nchains = nchains

        self.CD = CD
        self.l1= l1
        self.l1b = l1b
        self.l1c = l1c
        self.l1_custom = l1_custom
        self.l1b_custom = l1b_custom
        self.l2 = l2
        self.tmp_l2_fields = l2_fields
        self.tmp_reg_theta =reg_theta



        if self.N_PT>1:
            if record_acceptance==None:
                record_acceptance = True
            self.record_acceptance = record_acceptance

            if update_betas ==None:
                update_betas = True

            self._update_betas = update_betas

            if self.record_acceptance:
                self.mavar_gamma = 0.95
                self.acceptance_rates = np.zeros(N_PT-1,dtype=curr_float)
                self.mav_acceptance_rates = np.zeros(N_PT-1,dtype=curr_float)
            self.count_swaps = 0

            if self._update_betas:
                record_acceptance = True
                self.update_betas_lr = 0.1
                self.update_betas_lr_decay = self.decay_gamma

            if self._update_betas | (not hasattr(self,'betas')):
                self.betas =  np.arange(N_PT)/(N_PT-1)
                self.betas = self.betas[::-1].astype(curr_float)
            if (len(self.betas) != N_PT):
                self.betas =  np.arange(N_PT)/(N_PT-1)
                self.betas = self.betas[::-1].astype(curr_float)

            if bool(MoI) | bool(MoI_h):
                if MoI:
                    self.from_MoI = True
                    self.from_MoI_h = False
                    if type(MoI) == int:
                        MoI = moi.MoI(N=self.n_v, M = MoI , nature = self.visible)
                        if verbose | vverbose:
                            print('Fitting MOI first')
                        MoI.fit(data,weights=weights,verbose=0)
                        if verbose | vverbose:
                            print('Fitting MOI done')


                else:
                    self.from_MoI = False
                    self.from_MoI_h = True
                    update_params_MoI = True
                    if type(MoI_h) == int:
                        MoI_h = moi.MoI(N=self.n_h, M = MoI_h, nature = self.hidden)
                        if self.hidden == 'Potts':
                            MoI_h.weights = 0.01 * self.random_state.randn(MoI_h.M, self.n_h,self.n_ch).astype(curr_float)
                        else:
                            MoI_h.weights = 0.01 * self.random_state.randn(MoI_h.M, self.n_h).astype(curr_float)
                        MoI_h.cond_muv = self.hlayer.mean_from_inputs(MoI_h.weights)

                self.interpolate = interpolate
                if self.interpolate:
                    self.update_fields_lr = 0.1
                    self.update_fields_lr_decay = self.decay_gamma
                    if degree_interpolate is None:
                        if self.adapt_PT:
                            degree_interpolate = 5
                        else:
                            degree_interpolate = N_PT-1
                if self.from_MoI:
                    self.init_zlayer_portal(MoI,interpolate=interpolate,degree_interpolate=degree_interpolate,from_start=True,layer_id=0)
                    self._update_params_MoI = update_params_MoI
                else:
                    self.init_zlayer_portal(MoI_h,interpolate=interpolate,degree_interpolate=degree_interpolate,from_start=True,layer_id=1)
                    self._update_params_MoI = update_params_MoI

                if self._update_params_MoI:
                    self.update_MoI_lr = 1e-2
                    self.update_MoI_lr_decay = self.decay_gamma
            else:
                self.from_MoI= False
                self.from_MoI_h = False
                self.interpolate = False
                self._update_params_MoI = False
        else:
            self.from_MoI = False
            self.from_MoI_h = False
            self.interpolate= False
            self._update_betas = False
            self._update_params_MoI = False


        if self.N_PT > 1:
            self.fantasy_v = self.vlayer.random_init_config(self.nchains*self.N_PT).reshape([self.N_PT,self.nchains,self.vlayer.N])
            self.fantasy_h = self.hlayer.random_init_config(self.nchains*self.N_PT).reshape([self.N_PT,self.nchains,self.hlayer.N])
            if self.from_MoI | self.from_MoI_h:
                self.fantasy_z = self.zlayer.random_init_config(self.nchains*self.N_PT).reshape([self.N_PT,self.nchains,1])
            else:
                self.fantasy_E = np.zeros([self.N_PT,self.nchains],dtype=curr_float)
        else:
            if self.N_MC == 0:
                if self.from_hidden:
                    self.fantasy_h = utilities.make_all_discrete_configs(self.n_h,self.hidden,c=self.n_ch)
                    # self.fantasy_h = self.hlayer.random_init_config(self.nchains)
                else:
                    self.fantasy_v = utilities.make_all_discrete_configs(self.n_v,self.visible,c=self.n_cv)
            else:
                self.fantasy_v = self.vlayer.random_init_config(self.nchains)
                self.fantasy_h = self.hlayer.random_init_config(self.nchains)

        if ('TAU' in record) | self.adapt_MC:
            if MoI_tau is None:
                MoI_tau = MoI
            if self.N_PT>1:
                previous = np.asarray(np.argmax(MoI_tau.expectation(self.fantasy_v[0]),axis=-1)[:,np.newaxis],order='c')
            else:
                previous = np.asarray(np.argmax(MoI_tau.expectation(self.fantasy_v),axis=-1)[:,np.newaxis],order='c')
            joint_z = np.zeros([MoI_tau.M,MoI_tau.M],dtype=curr_float)
            smooth_jointz = 0.01**(1.0/record_interval)
            tau = 0


        if shuffle_data:
            if weights is not None:
                permute = np.arange(data.shape[0])
                self.random_state.shuffle(permute)
                weights = weights[permute]
                data = data[permute,:]
            else:
                self.random_state.shuffle(data)

        if weights is not None:
            weights = np.asarray(weights,dtype=curr_float)
            weights/=weights.mean()
        self.count_updates = 0
        if verbose:
            if weights is not None:
                lik = (self.pseudo_likelihood(data) * weights).sum()/weights.sum()
            else:
                lik = self.pseudo_likelihood(data).mean()
            print('Iteration number 0, pseudo-likelihood: %.2f'%lik)


        result = {}
        if 'weights' in record:
            result['weights'] = []
        if 'FV' in record:
            result['FV'] = []
        if 'FH' in record:
            result['FH'] = []
        if 'TH' in record:
            result['TH'] = []
        if 'B' in record:
            result['B'] = []
        if 'beta' in record:
            result['beta'] = []
        if 'p' in record:
            result['p'] = []
        if 'PL' in record:
            result['PL'] = []
        if 'PL_test' in record:
            result['PL_test'] = []
        if 'L' in record:
            result['L'] = []
        if 'L_test' in record:
            result['L_test'] = []
        if 'AP' in record:
            result['AP'] =[]
        if 'AM' in record:
            result['AM'] = []
        if 'A' in record:
            result['A'] = []
        if 'ETA' in record:
            result['ETA'] = []
        if 'AP0' in record:
            result['AP0'] = []
        if 'AM0' in record:
            result['AM0'] = []
        if 'AR' in record:
            result['AR'] = []
        if 'TAU' in record:
            result['TAU'] = []
        if 'theta' in record:
            result['theta'] = []


        count = 0
        n_iter = int(n_iter)
        for epoch in range(1,n_iter+1):
            if verbose:
                begin = time.time()
                tmp_prev_weights = self.weights.copy()
            if self.lr_decay:
                if (epoch>self.start_decay):
                    self.learning_rate*= self.decay_gamma
                    if self._update_betas:
                        self.update_betas_lr *= self.update_betas_lr_decay
                    if self.interpolate:
                        self.update_fields_lr *= self.update_fields_lr_decay
                    if self._update_params_MoI:
                        self.update_MoI_lr *= self.update_MoI_lr_decay
            if (verbose | vverbose):
                print('Starting epoch %s'%(epoch))
            for batch_slice in batch_slices:
                if weights is None:
                    self.minibatch_fit(data[batch_slice],weights=None)
                else:
                    self.minibatch_fit(data[batch_slice],weights=weights[batch_slice])


                if ('TAU' in record) | self.adapt_MC:
                    if self.N_PT>1:
                        current = np.asarray(np.argmax(MoI_tau.expectation(self.fantasy_v[0]),axis=-1)[:,np.newaxis],order='c')
                    else:
                        current = np.asarray(np.argmax(MoI_tau.expectation(self.fantasy_v),axis=-1)[:,np.newaxis],order='c')
                    joint_z = (1-smooth_jointz) * utilities.average_product(previous,current,c1=MoI_tau.M,c2=MoI_tau.M)[0,0] + smooth_jointz * joint_z
                    previous = current.copy()

                if self.adapt_PT:
                    if (count>0) & (count%update_every == 0):
                        curr_AR = self.mav_acceptance_rates.mean()/(1-self.mavar_gamma**(1+count) )
                        if  (curr_AR < self.AR_min) &  (self.N_PT < self.N_PT_max):
                            self.N_PT +=1
                            self.betas = np.concatenate( (self.betas * (self.N_PT-2)/(self.N_PT-1) + 1.0/(self.N_PT-1), self.betas[-1:] )  )
                            self.acceptance_rates = np.concatenate( (self.acceptance_rates, self.acceptance_rates[-1:]) )
                            self.mav_acceptance_rates = np.concatenate( (self.mav_acceptance_rates, self.mav_acceptance_rates[-1:]) )
                            self.fantasy_v = np.concatenate( (self.fantasy_v, self.fantasy_v[-1:]),axis=0)
                            self.fantasy_h = np.concatenate( (self.fantasy_h, self.fantasy_h[-1:]),axis=0)
                            if (self.from_MoI | self.from_MoI_h):
                                self.fantasy_z = np.concatenate( (self.fantasy_z, self.fantasy_z[-1:]),axis=0)
                            else:
                                self.fantasy_E = np.concatenate( (self.fantasy_E, self.fantasy_E[-1:]),axis=0)
                            if verbose | vverbose:
                                print('AR = %.3f, Increasing N_PT to %s'%(curr_AR,self.N_PT))

                if self.adapt_MC:
                    if (count>0) & (count%update_every == 0) :
                        Q = joint_z/(joint_z.sum(0)+1e-10)[np.newaxis,:]
                        lam,v  = np.linalg.eig(Q)
                        lam = lam[np.argsort(np.abs(np.real(lam)))[::-1]]
                        tau = -1/np.log(np.abs(np.real(lam[1])))
                        self.tau = tau
                        if (tau > self.tau_max) & (self.N_MC < self.N_MC_max):
                            self.N_MC +=1
                            if verbose | vverbose:
                                print('tau = %.2f, Increasing N_MC to %s'%(tau,self.N_MC))
                        elif (tau < (self.N_MC-1)/float(self.N_MC) * self.tau_max) & (self.N_MC>1):
                            self.N_MC -=1
                            if verbose | vverbose:
                                print('tau = %.2f, Decreasing N_MC to %s'%(tau,self.N_MC))




                if np.isnan(self.weights).max():
                    print('NAN in weights. Breaking')
                    return result



                if (count%record_interval ==0):
                    if 'weights' in record:
                        result['weights'].append( self.weights.copy() )
                    if 'FV' in record:
                        result['FV'].append(self.vlayer.fields.copy())
                    if 'FH' in record:
                        result['FH'].append(self.hlayer.fields.copy())
                    if 'TH' in record:
                        if self.hidden == 'ReLU':
                            result['TH'].append( (self.hlayer.theta_plus + self.hlayer.theta_minus)/(2*self.hlayer.a) )
                        elif self.hidden == 'ReLU+':
                            result['TH'].append( (self.hlayer.theta_plus - self.hlayer.mu_I)/self.hlayer.a)
                        else:
                            result['TH'].append(  (1-self.hlayer.eta**2)/2 * (self.hlayer.theta_plus + self.hlayer.theta_minus)/self.hlayer.a )
                    if 'PL' in record:
                        result['PL'].append( utilities.average(self.pseudo_likelihood(data),weights=weights) )
                    if 'PL_test' in record:
                        result['PL_test'].append( utilities.average(self.pseudo_likelihood(data_test),weights=weights_test) )
                    if 'L' in record:
                        if M_AIS>0:
                            N_PT = copy.copy(self.N_PT)
                            self.AIS(M=M_AIS,n_betas=n_betas_AIS,verbose=0,beta_type = 'linear')
                            self.N_PT = N_PT
                        else:
                            if (self.n_v <= self.n_h) | (self.hidden in ['Gaussian','ReLU','ReLU+','dReLU']):
                                tmp = False
                            else:
                                tmp = True
                            logZ,_,_,_ = self.compute_all_moments(from_hidden=tmp)
                            self.log_Z_AIS = logZ

                        result['L'].append(utilities.average(self.likelihood(data,recompute_Z=False),weights=weights))
                    if 'L_test' in record:
                        result['L_test'].append(utilities.average(self.likelihood(data_test,recompute_Z=False),weights=weights_test))
                    if 'beta' in record:
                        if self.n_cv >1:
                            result['beta'].append( (self.weights**2).sum(-1).sum(-1) )
                        else:
                            result['beta'].append( (self.weights**2).sum(-1) )
                    if 'p' in record:
                        if self.n_cv >1:
                            tmp = (self.weights**2).sum(-1)
                        else:
                            tmp = (self.weights**2)
                        a = 3
                        result['p'].append(  (tmp**a).sum(-1)**2/(tmp**(2*a)).sum(-1)/self.n_v  )
                    if 'AP' in record:
                        result['AP'].append(self.hlayer.a_plus.copy())
                    if 'AM' in record:
                        result['AM'].append(self.hlayer.a_minus.copy())
                    if 'A' in record:
                        result['A'].append(self.hlayer.a.copy())
                    if 'theta' in record:
                        result['theta'].append(self.hlayer.theta.copy())
                    if 'B' in record:
                        #result['B'].append(self.hlayer.b.copy())
                        result['B'].append(self.fantasy_v.mean(1))
                    if 'ETA' in record:
                        result['ETA'].append(self.hlayer.eta.copy())
                    if 'AP0' in record:
                        result['AP0'].append(self.hlayer.a_plus0.copy())
                    if 'AM0' in record:
                        result['AM0'].append(self.hlayer.a_minus0.copy())
                    if 'AR' in record:
                        result['AR'].append(self.mav_acceptance_rates.mean())
                    if 'TAU' in record:
                        Q = joint_z/(joint_z.sum(0)+1e-10)[np.newaxis,:]
                        lam,v  = np.linalg.eig(Q)
                        lam = lam[np.argsort(np.abs(np.real(lam)))[::-1]]
                        tau = -1/np.log(np.abs(np.real(lam[1])))
                        result['TAU'].append(tau.copy())



                count +=1

            if verbose:
                end = time.time()
                if weights is not None:
                    lik = (self.pseudo_likelihood(data) * weights).sum()/weights.sum()
                else:
                    lik = self.pseudo_likelihood(data).mean()
                message = "[%s] Iteration %d, time = %.2fs, pseudo-likelihood = %.2f"%(type(self).__name__, epoch,end - begin,lik)
                if self.N_PT>1:
                    AR = self.mav_acceptance_rates.mean()
                    message += ", AR = %.3f"%AR
                if ('TAU' in record) | self.adapt_MC:
                    message += ", tau = %.2f"%tau
                print(message)
                
                tmp_diff_weights = np.sum(np.abs(self.weights - tmp_prev_weights), axis=1)  # 1: per HU
                tmp_file = open('tmp_track_weights.txt', 'a')
                tmp_file.write(f'{tmp_diff_weights}\n')
                tmp_file.close()
                

            if shuffle_data:
                if weights is not None:
                    permute = np.arange(data.shape[0])
                    self.random_state.shuffle(permute)
                    weights = weights[permute]
                    data = data[permute,:]
                else:
                    self.random_state.shuffle(data)

        for key,item in result.items():
            result[key] = np.array(item)
        return result



    def minibatch_fit(self,V_pos,weights = None):
        self.count_updates +=1
        if self.CD: # Contrastive divergence: initialize the Markov chain at the data point.
            self.fantasy_v = V_pos
        # Else: use previous value.
        for _ in range(self.N_MC):
            if self.N_PT>1:
                if self.from_MoI:
                    (self.fantasy_v,self.fantasy_h,self.fantasy_z) = self.exchange_step_APT( (self.fantasy_v,self.fantasy_h,self.fantasy_z), record_acceptance=self.record_acceptance)
                    (self.fantasy_v,self.fantasy_h,self.fantasy_z) = self.markov_step_APT( (self.fantasy_v,self.fantasy_h,self.fantasy_z), beta = self.betas, recompute=False)
                elif self.from_MoI_h:
                    (self.fantasy_v,self.fantasy_h,self.fantasy_z) = self.exchange_step_APTh( (self.fantasy_v,self.fantasy_h,self.fantasy_z), record_acceptance=self.record_acceptance)
                    (self.fantasy_v,self.fantasy_h,self.fantasy_z) = self.markov_step_APTh( (self.fantasy_v,self.fantasy_h,self.fantasy_z) , beta = self.betas, recompute=False)
                else:
                    (self.fantasy_v,self.fantasy_h),self.fantasy_E = self.markov_step_and_energy((self.fantasy_v,self.fantasy_h),self.fantasy_E, beta = self.betas)
                    (self.fantasy_v,self.fantasy_h),self.fantasy_E = self.exchange_step_PT((self.fantasy_v,self.fantasy_h),self.fantasy_E,record_acceptance=self.record_acceptance,compute_energy=False)
            else:
                (self.fantasy_v,self.fantasy_h) = self.markov_step((self.fantasy_v,self.fantasy_h) )

        if self.N_PT>1:
            if self._update_betas:
                self.update_betas()
            if self._update_params_MoI:
                if self.from_MoI:
                    self.update_moments_MoI(self.fantasy_v[0],layer_id=0,eps= 1e-5)
                    self.update_params_MoI(layer_id =0 , eps = 1e-5)
                else:
                    self.update_moments_MoI(self.fantasy_h[0],layer_id=1,eps= 1e-5)
                    self.update_params_MoI(layer_id =1 , eps = 1e-5)
            if self.interpolate:
                self.update_interpolation_MoI(self.fantasy_z,self.betas,self.update_fields_lr)

        if self.N_PT>1:
            V_neg = self.fantasy_v[0,:,:]
        else:
            if self.from_hidden:
                V_neg = self.mean_visibles(self.fantasy_h)
                H_neg = self.fantasy_h
            else:
                V_neg = self.fantasy_v

        if self.N_MC>0: # No Monte Carlo. Compute exhaustively the moments using all 2**N configurations.
            weights_neg = None
        else:
            if self.from_hidden:
                F = self.free_energy_h(H_neg)
            else:
                F = self.free_energy(V_neg)
            F -= F.min()
            weights_neg = np.exp(-F)
            weights_neg /= weights_neg.sum()
        I_pos = self.vlayer.compute_output(V_pos,self.weights)
        if not self.from_hidden:
            I_neg = self.vlayer.compute_output(V_neg,self.weights)

        if self.batch_norm:
            if (self.n_cv >1) & (self.n_ch  == 1):
                mu_I = np.tensordot(self.weights,self.mu_data,axes = [(1,2),(0,1)])
            elif (self.n_cv >1) & (self.n_ch  > 1):
                mu_I = np.tensordot(self.weights,self.mu_data,axes = [(1,3),(0,1)])
            elif (self.n_cv  == 1) & (self.n_ch  > 1):
                mu_I = np.tensordot(self.weights,self.mu_data,axes = [1,0])
            else:
                mu_I = np.dot(self.weights,self.mu_data)


            self.hlayer.batch_norm_update(mu_I,I_pos,lr= self.learning_rate/self.learning_rate_init,weights=weights)

        H = self.hlayer.mean_from_inputs(I_pos)
        if self.from_hidden:
            self.gradient['vlayer'] = self.vlayer.internal_gradients(self.mu_data[np.newaxis],V_neg,weights=None,weights_neg=weights_neg,value='mean')
            self.gradient['hlayer'] = self.hlayer.internal_gradients(H,H_neg,weights=weights,weights_neg=weights_neg,value='mean',value_neg='data')
            self.gradient['weights'] = pgm.couplings_gradients_h(self.weights,H,H_neg,V_pos,V_neg,self.n_ch, self.n_cv, l1 = self.l1, l1b = self.l1b, l1c = self.l1c, l2 = self.l2,weights=weights,weights_neg=weights_neg,l1_custom=self.l1_custom,l1b_custom=self.l1b_custom)

        else:
            H_neg = self.hlayer.mean_from_inputs(I_neg)
            self.gradient['vlayer'] = self.vlayer.internal_gradients(V_pos,V_neg,weights=weights,weights_neg=weights_neg,value='data')
            self.gradient['hlayer'] = self.hlayer.internal_gradients(I_pos,I_neg,weights=weights,weights_neg=weights_neg,value='input')
            self.gradient['weights'] = pgm.couplings_gradients(self.weights,H,H_neg,V_pos,V_neg,self.n_ch, self.n_cv, mean1 = True, l1 = self.l1, l1b = self.l1b, l1c = self.l1c, l2 = self.l2,weights=weights,weights_neg=weights_neg,l1_custom=self.l1_custom,l1b_custom=self.l1b_custom)

        if self.batch_norm: # Modify gradients.
            self.hlayer.batch_norm_update_gradient(self.gradient['weights'], self.gradient['hlayer'],V_pos,I_pos, self.mu_data,self.n_cv,weights=weights)

        for layer_ in ['vlayer','hlayer']:
            for internal_param,gradient in self.gradient[layer_].items():
                saturate(gradient,1.0)
        saturate(self.gradient['weights'],1.0)


        if self.tmp_l2_fields>0:
            self.gradient['vlayer']['fields'] -= self.tmp_l2_fields *  self.vlayer.fields
        if not self.tmp_reg_theta ==0:
            self.gradient['hlayer']['theta'] -= self.tmp_reg_theta

        for layer_ in ['vlayer','hlayer']:
            for internal_param,gradient in self.gradient[layer_].items():
                if self.do_grad_updates[layer_][internal_param]:
                    current = getattr(getattr(self,layer_),internal_param)
                    if self.optimizer == 'SGD':
                        current += self.learning_rate * gradient
                    elif self.optimizer == 'momentum':
                        self.previous_update[layer_][internal_param] =  (1- self.momentum) * self.learning_rate * gradient + self.momentum * self.previous_update[layer_][internal_param]
                        current += self.previous_update[layer_][internal_param]
                    elif self.optimizer == 'ADAM':
                        self.gradient_moment1[layer_][internal_param] *= self.beta1
                        self.gradient_moment1[layer_][internal_param] += (1- self.beta1) * gradient
                        self.gradient_moment2[layer_][internal_param] *= self.beta2
                        self.gradient_moment2[layer_][internal_param] += (1- self.beta2) * gradient**2
                        current += self.learning_rate * (self.gradient_moment1[layer_][internal_param]/(1-self.beta1**self.count_updates)) /(self.epsilon + np.sqrt( self.gradient_moment2[layer_][internal_param]/(1-self.beta2**self.count_updates ) ) )


        if self.do_grad_updates['weights']:
            if self.optimizer == 'SGD':
                self.weights += self.learning_rate * self.gradient['weights']
            elif self.optimizer == 'momentum':
                self.previous_update['weights'] = (1-self.momentum) * self.learning_rate * self.gradient['weights'] + self.momentum * self.previous_update['weights']
                self.weights += self.previous_update['weights']
            elif self.optimizer == 'ADAM':
                self.gradient_moment1['weights'] *= self.beta1
                self.gradient_moment1['weights'] += (1-self.beta1) * self.gradient['weights']
                self.gradient_moment2['weights'] *= self.beta2
                self.gradient_moment2['weights'] += (1-self.beta2) * self.gradient['weights']**2
                self.weights += self.learning_rate * (self.gradient_moment1['weights']/(1-self.beta1**self.count_updates))/(self.epsilon + np.sqrt( self.gradient_moment2['weights']/(1-self.beta2**self.count_updates  )))

            if (self.n_cv>1) | (self.n_ch >1):
                pgm.gauge_adjust_couplings(self.weights,self.n_ch,self.n_cv,gauge=self.gauge)
            self.hlayer.recompute_params()



    def initialize_gradient_dictionary(self):
        out = {}
        out['vlayer'] = self.vlayer.internal_gradients(np.zeros([1,self.n_v],dtype=self.vlayer.type), np.zeros([1,self.n_v],dtype=self.vlayer.type) )
        out['hlayer'] = self.hlayer.internal_gradients(np.zeros([1,self.n_h],dtype=self.hlayer.type), np.zeros([1,self.n_h],dtype=self.hlayer.type) )
        out['weights'] = np.zeros_like(self.weights)
        return out
