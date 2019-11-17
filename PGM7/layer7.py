import sys
import numpy as np
from scipy.sparse import csr_matrix
import numba_utilities7 as cy_utilities
import batch_norm_utils7 as batch_norm_utils
from utilities7 import check_random_state,logistic,softmax,invert_softmax,cumulative_probabilities,erf_times_gauss,log_erf_times_gauss,logsumexp,average,bilinear_form,average_product,covariance,add_to_gradient,saturate, erf, erfinv, reshape_in,reshape_out

double_precision = False

if double_precision:
    curr_float = np.float64
    curr_int = np.int64
else:
    curr_float = np.float32
    curr_int = np.int16

#%% Layer class

def initLayer(nature='Bernoulli', **kwargs):
    exec('layer=%sLayer(**kwargs)'%nature)
    return locals()['layer']




class Layer():
    def __init__(self, N = 100, nature = 'Bernoulli', position = 'visible',batch_norm=False, n_c = 1, random_state = None):
        self.N = N
        self.nature = nature
        self.random_state = check_random_state(random_state)
        self.position = position
        self.n_c = n_c
        self.previous_beta = None
        if self.nature in ['Bernoulli','Gaussian','ReLU','dReLU','ReLU+']:
            self.type= curr_float
        else:
            self.type = curr_int

        self.batch_norm = batch_norm
        if self.position == 'hidden':
            if self.n_c > 1:
                self.mu_I = np.zeros([N,n_c],dtype=curr_float) # For batch_norm.
            else:
                self.mu_I = np.zeros(N,dtype=curr_float) # For batch_norm.

        if self.nature in ['Gaussian','ReLU','dReLU','ReLU+']: # For batch_norm.
            self.gamma_min = 0.05
            self.gamma_drop_max = 0.75


    def get_input(self,I,I0=None,beta=1):
        if I is None:
            if self.n_c>1:
                I = np.zeros([1,self.N,self.n_c], dtype=curr_float)
            else:
                I = np.zeros([1,self.N], dtype=curr_float)

        if type(beta) == np.ndarray:
            if self.n_c>1:
                beta = beta[:,np.newaxis,np.newaxis,np.newaxis]
            else:
                beta = beta[:,np.newaxis,np.newaxis]
            beta_not_one = True
        else:
            beta_not_one = (beta !=1)

        if beta_not_one:
            I = I*beta
            if I0 is not None:
                I = I+(1-beta) * I0
        if I.ndim == 1: I = I[np.newaxis, :] # ensure that the config data is a batch, at least of just one vector
        return I

    def get_params(self,beta=1):
        if type(beta) == np.ndarray:
            if self.n_c>1:
                beta = beta[:,np.newaxis,np.newaxis,np.newaxis]
            else:
                beta = beta[:,np.newaxis,np.newaxis]
            beta_is_array = True
            beta_not_one = True
        else:
            beta_not_one = (beta !=1)
            beta_is_array = False


        for key in self.list_params:
            if beta_not_one & self.params_anneal[key]:
                tmp = beta * getattr(self,key) + (1-beta) * getattr(self,key + '0')
            else:
                tmp = getattr(self,key)
            if self.params_newaxis[key] and not beta_is_array:
                tmp = tmp[np.newaxis,:]
            setattr(self,'_' + key, tmp)

    def compute_output(self,config, couplings, direction='up',out=None):
        case1 = (self.n_c>1) & (couplings.ndim == 4) # output layer is Potts
        case2 = (self.n_c>1) & (couplings.ndim == 3)
        case3 = (self.n_c==1) & (couplings.ndim == 3) # output layer is Potts
        case4 = (self.n_c==1) & (couplings.ndim == 2)

        if direction == 'up':
            N_output_layer = couplings.shape[0]
            if case1 | case3:
                n_c_output_layer = couplings.shape[2]
            else:
                n_c_output_layer = 1
        else:
            N_output_layer = couplings.shape[1]
            if case1 | case3:
                n_c_output_layer = couplings.shape[-1]
            else:
                n_c_output_layer = 1


        config,xshape = reshape_in(config,xdim=1)

        if case1 | case3:
            out_dim = list(xshape[:-1]) + [N_output_layer,n_c_output_layer]
            out_ndim = 2
        else:
            out_dim = list(xshape[:-1]) + [N_output_layer]
            out_ndim = 1
        if out is not None:
            if not list(out.shape) == out_dim:
                print('Mismatch dimensions %s, %s, reinitializating I'%(out.shape, out_dim))
                out = np.empty(out_dim,dtype=curr_float)
            out,_ = reshape_in(out,xdim=out_ndim)

        if direction == 'up':
            if case1:
                out =  cy_utilities.compute_output_Potts_C(self.N, self.n_c,n_c_output_layer ,config,couplings)
            elif case2:
                out = cy_utilities.compute_output_C(self.N, self.n_c,config,couplings)
            elif case3:
                out = np.dot(config, couplings, out=out)
            else:
                out = np.dot(config, couplings.T,out=out)
        elif direction == 'down':
            if case1:
                out = cy_utilities.compute_output_Potts_C2(self.N, self.n_c,n_c_output_layer ,config, couplings)
            elif case2:
                out = cy_utilities.compute_output_C2(self.N, self.n_c,config,couplings)
            elif case3:
                out = np.tensordot(config, couplings, axes = (-1,0) )
            else:
                out = np.dot(config, couplings,out=out)
        return reshape_out(out,xshape,xdim=1)

    def logpartition(self,I,I0=None,beta = 1):
        return self.cgf_from_inputs(I,I0=I0,beta=beta).sum(-1)

    def random_init_config(self,n_samples,N_PT=1):
        if not 'coupled' in self.nature:
            if self.n_c ==1:
                if N_PT>1:
                    return self.sample_from_inputs( np.zeros([N_PT*n_samples, self.N],dtype=curr_float) ,beta = 0).reshape([N_PT,n_samples,self.N])
                else:
                    return self.sample_from_inputs( np.zeros([n_samples, self.N],dtype=curr_float) ,beta = 0).reshape([n_samples,self.N])
            else:
                if N_PT>1:
                    return self.sample_from_inputs( np.zeros([N_PT*n_samples, self.N,self.n_c],dtype=curr_float) ,beta = 0).reshape([N_PT,n_samples,self.N])
                else:
                    return self.sample_from_inputs( np.zeros([n_samples, self.N,self.n_c],dtype=curr_float) ,beta = 0).reshape([n_samples,self.N])

        elif self.nature in ['Bernoulli_coupled','Spin_coupled']:
            if N_PT>1:
                (x,fields_eff) = self.sample_from_inputs( np.zeros([N_PT*n_samples, self.N],dtype=curr_float) ,beta = 0)
                x = x.reshape([N_PT,n_samples,self.N])
                fields_eff = fields_eff.reshape([N_PT,n_samples,self.N])
            else:
                (x,fields_eff) = self.sample_from_inputs( np.zeros([N_PT*n_samples, self.N],dtype=curr_float) ,beta = 0)
            return (x,fields_eff)
        elif self.nature == 'Potts_coupled':
            if N_PT>1:
                (x,fields_eff) = self.sample_from_inputs( np.zeros([n_samples*N_PT, self.N, self.n_c],dtype=curr_float) ,beta = 0)
                x = x.reshape([N_PT,n_samples,self.N])
                fields_eff = fields_eff.reshape([N_PT,n_samples,self.N,self.n_c])
            else:
                (x,fields_eff) = self.sample_from_inputs( np.zeros([n_samples, self.N, self.n_c],dtype=curr_float) ,beta = 0)
            return (x,fields_eff)

    def sample_and_energy_from_inputs(self,I,I0=None,beta=1,previous=(None,None),remove_init=False):
        if not 'coupled' in self.nature:
            config = self.sample_from_inputs(I,beta=beta,I0=I0,previous=previous)
            if remove_init:
                if I0 is not None:
                    I = I - I0
            else:
                I = self.get_input(I,I0=I0,beta=beta)

            energy = self.energy(config,remove_init=remove_init,beta=beta)
            if self.n_c==1:
                energy -= (I * config).sum(1)
            else:
                I,Idim = reshape_in(I,xdim=2)
                energy -= reshape_out(cy_utilities.dot_Potts2_C(self.N, self.n_c,reshape_in(config,xdim=1)[0], I), Idim, xdim=2)
            return (config, energy)
        else:
            (x,fields_eff) = self.sample_from_inputs(I,I0=I0,beta=beta,previous=previous)
            if remove_init:
                f = 0.5* (self.fields[np.newaxis] + fields_eff) - self.fields0[np.newaxis]
                if I is not None:
                    f += I
                if I0 is not None:
                    f -= I0
            else:
                f = beta * fields_eff + (1-beta) * self.fields0[np.newaxis]
                if I is not None:
                    f += beta * I
                if I0 is not None:
                    f += (1-beta) * I0
            if self.nature =='Potts_coupled':
                I,Idim = reshape_in(I,xdim=2)
                energy = -reshape_out( cy_utilities.dot_Potts2_C(self.N, self.n_c,reshape_in(x,xdim=1)[0], f) , Idim, xdim=2)
            else:
                energy = -np.sum(x*f,-1)
            return (x,fields_eff),energy

    def recompute_params(self,which='regular'): # relevant when there is a change of variable.
        return




class BernoulliLayer(Layer):
    def __init__(self,N=100,position='visible', random_state = None,batch_norm=False,zero_field = False,**kwargs):
        super(BernoulliLayer, self).__init__(N = N, nature='Bernoulli',position=position, batch_norm=batch_norm, n_c=1,random_state=random_state)
        self.zero_field = zero_field
        self.fields = np.zeros(self.N,dtype=curr_float)
        self.fields0 = np.zeros(self.N,dtype=curr_float) # useful for PT.
        self.list_params = ['fields']
        self.params_anneal = {'fields':True}
        self.params_newaxis = {'fields':True}
        self.do_grad_updates = {'fields':~self.zero_field, 'fields0': (~self.zero_field) & (self.position =='hidden')}
        self.do_grad_updates_batch_norm = self.do_grad_updates

    def mean_from_inputs(self,I,I0=None,beta=1):
        I = self.get_input(I,I0=I0,beta=beta)
        self.get_params(beta=beta)
        return logistic(I+self._fields)

    def var_from_inputs(self,I,I0=None,beta=1):
        mean = self.mean_from_inputs(I,I0=I0,beta=beta)
        return mean * (1-mean)

    def cgf_from_inputs(self,I,I0=None,beta=1):
        I = self.get_input(I,I0=I0,beta=beta)
        self.get_params(beta=beta)
        return np.logaddexp(0, self._fields + I)

    def transform(self,I):
        self.get_params(beta=1)
        return (I + self._fields)>0

    def sample_from_inputs(self,I,I0=None,beta=1,out=None,**kwargs):
        I = self.get_input(I,I0=I0,beta=beta)
        self.get_params(beta=beta)
        if out is None:
            out = np.empty_like(I)
        if type(beta) == np.ndarray:
            cy_utilities.sample_from_inputs_Bernoulli_numba3(I,self._fields[:,0,:],out)
        else:
            cy_utilities.sample_from_inputs_Bernoulli_numba2(I,self._fields[0,:],out)
        return out


    def energy(self,config,remove_init = False,beta=1):
        if remove_init:
            return -np.dot(config, self.fields - self.fields0)
        else:
            self.get_params(beta=beta)
            return -(config * self._fields).sum(-1)

    def internal_gradients(self,data_pos,data_neg,weights = None,weights_neg=None,value='data',value_neg=None,**kwargs):
        gradients = {}
        if value_neg is None:
            value_neg = value
        if value == 'input':
            data_pos = self.mean_from_inputs(data_pos)
        mu_pos = average(data_pos,weights=weights)
        if value_neg == 'input':
            data_neg = self.mean_from_inputs(data_neg)
        mu_neg = average(data_neg,weights=weights_neg)
        gradients['fields'] =   mu_pos - mu_neg
        if weights is not None:
            gradients['fields'] *= weights.mean()
        return gradients

    def init_params_from_data(self,X,eps=1e-6,mean=False,weights=None):
        if X is None:
            self.fields = np.zeros(self.N,dtype=curr_float)
            self.fields0 = np.zeros(self.N,dtype=curr_float)
        else:
            if mean:
                mu = X
            else:
                mu = average(X,weights=weights)
            self.fields = np.log((mu+ eps)/(1-mu + eps))
            self.fields0 = self.fields.copy()

    def batch_norm_update(self,mu_I,I,**kwargs):
        delta_mu_I = (mu_I-self.mu_I)
        self.mu_I = mu_I
        self.fields -= delta_mu_I


    def batch_norm_update_gradient(self,gradient_W, gradient_hlayer,V,I,mu,n_c,weights=None):
        add_to_gradient(gradient_W, gradient_hlayer['fields'], -mu)
        return




class SpinLayer(Layer):
    def __init__(self,N=100,position='visible', batch_norm=False, random_state = None,zero_field = False,**kwargs):
        super(SpinLayer, self).__init__(N = N, nature='Spin',position=position, batch_norm=batch_norm, n_c=1,random_state=random_state)
        self.zero_field = zero_field
        self.fields = np.zeros(self.N,dtype=curr_float)
        self.fields0 = np.zeros(self.N,dtype=curr_float) # useful for PT.
        self.list_params = ['fields']
        self.params_anneal = {'fields':True}
        self.params_newaxis = {'fields':True}
        self.do_grad_updates = {'fields':~self.zero_field, 'fields0': (~self.zero_field) & (self.position =='hidden')}
        self.do_grad_updates_batch_norm = self.do_grad_updates


    def mean_from_inputs(self,I,I0=None,beta=1):
        I = self.get_input(I,I0=I0,beta=beta)
        self.get_params(beta=beta)
        return np.tanh(I+self._fields)

    def var_from_inputs(self,I,I0=None,beta=1):
        return 1- self.mean_from_inputs(I,I0=I0,beta=beta)**2

    def cgf_from_inputs(self,I,I0=None,beta=1):
        I = self.get_input(I,I0=I0,beta=beta)
        self.get_params(beta=beta)
        tmp = self._fields + I
        return np.logaddexp(-tmp,tmp)

    def transform(self,I):
        self.get_params(beta=1)
        return np.sign(I + self.fields)

    def sample_from_inputs(self,I,I0=None,beta=1,out=None,**kwargs):
        I = self.get_input(I,I0=I0,beta=beta)
        self.get_params(beta=beta)
        if out is None:
            out = np.empty_like(I)
        if type(beta) == np.ndarray:
            cy_utilities.sample_from_inputs_Spin_numba3(I,self._fields[:,0,:],out)
        else:
            cy_utilities.sample_from_inputs_Spin_numba2(I,self._fields[0,:],out)
        return out

    def energy(self,config,remove_init = False,beta=1):
        if remove_init:
            return -np.dot(config, self.fields - self.fields0)
        else:
            self.get_params(beta=beta)
            return -(config * self._fields).sum(-1)

    def internal_gradients(self,data_pos,data_neg,weights = None,weights_neg=None,value='data',value_neg=None,**kwargs):
        gradients = {}
        if value_neg is None:
            value_neg = value
        if value == 'input':
            data_pos = self.mean_from_inputs(data_pos)
        mu_pos = average(data_pos,weights=weights)
        if value_neg == 'input':
            data_neg = self.mean_from_inputs(data_neg)
        mu_neg = average(data_neg,weights=weights_neg)
        gradients['fields'] =   mu_pos - mu_neg
        if weights is not None:
            gradients['fields'] *= weights.mean()
        return gradients

    def init_params_from_data(self,X,eps=1e-6,mean=False,weights=None):
        if X is None:
            self.fields = np.zeros(self.N,dtype=curr_float)
            self.fields0 = np.zeros(self.N,dtype=curr_float)
        else:
            if mean:
                mu = X
            else:
                mu = average(X,weights=weights)
            self.fields= 0.5*np.log((1+mu + eps)/(1-mu + eps) )
            self.fields0 = self.fields.copy()

    def batch_norm_update(self,mu_I,I,**kwargs):
        delta_mu_I = (mu_I-self.mu_I)
        self.mu_I = mu_I
        self.fields -= delta_mu_I

    def batch_norm_update_gradient(self,gradient_W, gradient_hlayer,V,I,mu,n_c,weights=None):
        add_to_gradient(gradient_W, gradient_hlayer['fields'], -mu)
        return




class PottsLayer(Layer):
    def __init__(self,N=100,position='visible', batch_norm=False,gauge='zerosum', n_c=2, random_state = None,zero_field = False,**kwargs):
        super(PottsLayer, self).__init__(N = N, nature='Potts',position=position, batch_norm=batch_norm, n_c=n_c,random_state=random_state)
        self.zero_field = zero_field
        self.fields = np.zeros([self.N,self.n_c],dtype=curr_float)
        self.fields0 = np.zeros([self.N,self.n_c],dtype=curr_float)
        self.gauge = gauge
        self.list_params = ['fields']
        self.params_anneal = {'fields':True}
        self.params_newaxis = {'fields':True}
        self.do_grad_updates = {'fields':~self.zero_field, 'fields0': (~self.zero_field) & (self.position =='hidden')}
        self.do_grad_updates_batch_norm = self.do_grad_updates


    def mean_from_inputs(self,I,I0=None,beta=1):
        I = self.get_input(I,I0=I0,beta=beta)
        self.get_params(beta=beta)
        return softmax(I + self._fields)

    def var_from_inputs(self,I,I0=None,beta=1):
        self.get_params(beta=beta)
        mean = self.mean_from_inputs(I,I0=I0,beta=beta)
        return mean * (1-mean)

    def cgf_from_inputs(self,I,I0=None,beta=1):
        I = self.get_input(I,I0=I0,beta=beta)
        self.get_params(beta=beta)
        return logsumexp(self._fields+ I,-1)

    def transform(self,I):
        self.get_params(beta=1)
        return np.argmax(I + self._fields,axis=-1)

    def sample_from_inputs(self,I,I0=None,beta=1,out=None,**kwargs):
        I = self.get_input(I,I0=I0,beta=beta)
        self.get_params(beta=beta)
        if out is None:
            out = np.empty(I.shape[:-1],dtype=self.type)
        if type(beta) == np.ndarray:
            cy_utilities.sample_from_inputs_Potts_numba3(I,self._fields[:,0,:,:],out)
        else:
            cy_utilities.sample_from_inputs_Potts_numba2(I,self._fields[0,:,:],out)
        return out

    def energy(self,config,remove_init = False,beta=1):
        if remove_init:
            fields = self.fields - self.fields0
        else:
            if beta!=1:
                fields = beta* self.fields + (1-beta) * self.fields0
            else:
                fields = self.fields
        config,dim = reshape_in(config,xdim=1)
        return reshape_out(-cy_utilities.dot_Potts_C(self.N, self.n_c,config, fields), dim,xdim=1)

    def internal_gradients(self,data_pos,data_neg,weights = None,weights_neg=None,value='data',value_neg=None,**kwargs):
        gradients = {}
        if value_neg is None:
            value_neg = value
        if value == 'data':
            mu_pos = average(data_pos,weights=weights, c = self.n_c)
        elif value == 'mean':
            mu_pos = average(data_pos,weights=weights)
        elif value == 'input':
            mu_pos = average(self.mean_from_inputs(data_pos),weights=weights)

        if value_neg == 'data':
            mu_neg = average(data_neg,weights=weights_neg, c = self.n_c)
        elif value_neg == 'mean':
            mu_neg = average(data_neg,weights=weights_neg)
        elif value_neg == 'input':
            mu_neg = average(self.mean_from_inputs(data_neg),weights=weights_neg)

        gradients['fields'] =   mu_pos - mu_neg
        if weights is not None:
            gradients['fields'] *= weights.mean()
        return gradients

    def init_params_from_data(self,X,eps=1e-6,mean=False,weights=None):
        if X is None:
            self.fields = np.zeros([self.N,self.n_c],dtype=curr_float)
            self.fields0 = np.zeros([self.N,self.n_c],dtype=curr_float)
        else:
            if mean:
                mu = X
            else:
                mu = average(X,weights=weights,c=self.n_c)
            self.fields = invert_softmax(mu,eps=eps, gauge = self.gauge)
            self.fields0 = self.fields.copy()

    def batch_norm_update(self,mu_I,I,**kwargs):
        delta_mu_I = (mu_I-self.mu_I)
        self.mu_I = mu_I
        self.fields -= delta_mu_I

    def batch_norm_update_gradient(self,gradient_W, gradient_hlayer,V,I,mu,n_c,weights=None):
        add_to_gradient(gradient_W, gradient_hlayer['fields'], -mu)
        return





class GaussianLayer(Layer):
    def __init__(self,N=100,position='visible', batch_norm=False, random_state = None,**kwargs):
        super(GaussianLayer, self).__init__(N = N, nature='Gaussian',position=position, batch_norm=batch_norm,n_c=1,random_state=random_state)
        self.gamma = np.ones(self.N,dtype=curr_float)
        self.gamma0 = np.ones(self.N,dtype=curr_float)
        self.theta = np.zeros(self.N,dtype=curr_float)
        self.theta0 = np.zeros(self.N,dtype=curr_float)
        self.list_params = ['gamma','theta']
        self.params_anneal = {'gamma':True,'theta':True}
        self.params_newaxis = {'gamma':True,'theta':True}
        if self.position == 'visible':
            self.do_grad_updates = {'gamma':True,'theta':True,'gamma0':False,'theta0':False}
        else:
            self.do_grad_updates = {'gamma':False,'theta':False,'gamma0':True,'theta0':True}
        self.do_grad_updates_batch_norm = {'gamma':False,'theta':False,'gamma0':False,'theta0':False}

    def mean_from_inputs(self,I,I0=None,beta=1):
        I = self.get_input(I,I0=I0,beta=beta)
        self.get_params(beta=beta)
        return (I - self._theta)/self._gamma

    def mean2_from_inputs(self,I,I0=None,beta=1):
        I = self.get_input(I,I0=I0,beta=beta)
        self.get_params(beta=beta)
        return self.mean_from_inputs(I,I0=I0,beta=beta)**2 + 1/self._gamma

    def var_from_inputs(self,I,I0=None,beta=1):
        self.get_params(beta=beta)
        return np.ones(I.shape)/self._gamma

    def cgf_from_inputs(self,I,I0=None,beta=1):
        I = self.get_input(I,I0=I0,beta=beta)
        self.get_params(beta=beta)
        return (0.5 * (I - self._theta)**2/self._gamma) + 0.5 * np.log(2*np.pi/self._gamma)

    def transform(self,I):
        return self.mean_from_inputs(I)


    def sample_from_inputs(self,I,I0=None,beta=1,out=None,**kwargs):
        I = self.get_input(I,I0=I0,beta=beta)
        self.get_params(beta=beta)
        if out is None:
            out = np.empty_like(I)
        out = (I - self._theta)/self._gamma + self.random_state.randn(*I.shape).astype(curr_float)/np.sqrt(self._gamma)
        return out

    def energy(self,config,remove_init = False,beta=1):
        if remove_init:
            return np.dot(config**2, self.gamma - self.gamma0)/2 + np.dot(config, self.theta - self.theta0)
        else:
            self.get_params(beta=beta)
            return (config**2 * self._gamma).sum(-1)/2 + (config * self._theta).sum(-1)

    def internal_gradients(self,data_pos,data_neg,weights = None,weights_neg=None,value='data',value_neg=None,**kwargs):
        gradients = {}
        if value_neg is None:
            value_neg = value
        if value == 'data':
            mu2_pos = average(data_pos**2,weights=weights)
            mu_pos = average(data_pos,weights=weights)
        elif value == 'mean':
            print('gaussian mean not supported for internal gradient')
        elif value == 'input':
            mu2_pos = average(self.mean2_from_inputs(data_pos),weights=weights)
            mu_pos = average(self.mean_from_inputs(data_pos),weights=weights)
        if value_neg == 'data':
            mu2_neg = average(data_neg**2,weights=weights_neg)
            mu_neg = average(data_neg,weights=weights_neg)
        elif value_neg == 'mean':
            print('gaussian mean not supported for internal gradient')
        elif value_neg == 'input':
            mu2_neg = average(self.mean2_from_inputs(data_neg),weights=weights_neg)
            mu_neg = average(self.mean_from_inputs(data_neg),weights=weights_neg)

        gradients['gamma'] = -0.5 * (mu2_pos - mu2_neg)
        gradients['theta'] = -mu_pos+mu_neg

        if weights is not None:
            gradients['gamma'] *= weights.mean()
            gradients['theta'] *= weights.mean()
        return gradients

    def init_params_from_data(self,X,eps=1e-6,mean=False,weights=None):
        if X is None:
            self.gamma = np.ones(self.N,dtype=curr_float)
            self.gamma0 = np.ones(self.N,dtype=curr_float)
            self.theta = np.zeros(self.N,dtype=curr_float)
            self.theta0 = np.zeros(self.N,dtype=curr_float)
        else:
            mu = average(X,weights=weights)
            var = average(X**2,weights=weights) - mu**2
            self.gamma = 1/(var+eps)
            self.theta = - self.gamma * mu
            self.gamma0 = self.gamma.copy()
            self.theta0 = self.theta.copy()

    def batch_norm_update(self,mu_I,I,lr=1,weights=None):
        delta_mu_I = (mu_I-self.mu_I)
        self.mu_I = mu_I
        self.theta += delta_mu_I
        var_e = average(I**2,weights=weights) - average(I,weights=weights)**2
        new_gamma = (1+np.sqrt(1+4*var_e))/2
        self.gamma = np.maximum( self.gamma_min,
        np.maximum(
        (1-lr) * self.gamma + lr * new_gamma,
        self.gamma_drop_max * self.gamma
        )
        )

    def batch_norm_update_gradient(self,gradient_W, gradient_hlayer,V,I,mu,n_c,weights=None):
        WChat = covariance(I,V,weights=weights,c1=1,c2=n_c)
        var_e = average(I**2,weights=weights) - average(I,weights=weights)**2
        if n_c >1:
            dgamma_dw = 2/np.sqrt(1+ 4 * var_e)[:,np.newaxis,np.newaxis] * WChat
        else:
            dgamma_dw = 2/np.sqrt(1+ 4 * var_e)[:,np.newaxis] * WChat

        add_to_gradient(gradient_W, gradient_hlayer['theta'], mu)
        add_to_gradient(gradient_W, gradient_hlayer['gamma'], dgamma_dw)
        return


class ReLUplusLayer(Layer):
    def __init__(self,N=100,position='visible', batch_norm=False, random_state = None,**kwargs):
        super(ReLUplusLayer, self).__init__(N = N, nature='ReLUplus',position=position, batch_norm=batch_norm, n_c=1,random_state=random_state)
        self.gamma = np.ones(self.N,dtype=curr_float)
        self.gamma0 = np.ones(self.N,dtype=curr_float)
        self.theta = np.zeros(self.N,dtype=curr_float)
        self.theta0 = np.zeros(self.N,dtype=curr_float)
        self.list_params = ['gamma','theta']
        self.params_anneal = {'gamma':True,'theta':True}
        self.params_newaxis = {'gamma':True,'theta':True}
        if self.position == 'visible':
            self.do_grad_updates = {'gamma':True,'theta':True,'gamma0':False,'theta0':False}
        else:
            self.do_grad_updates = {'gamma':False,'theta':True,'gamma0':True,'theta0':True}
        self.do_grad_updates_batch_norm = {'gamma':False,'theta':True,'gamma0':False,'theta0':False}

    def mean_from_inputs(self,I,I0=None,beta=1):
        I = self.get_input(I,I0=I0,beta=beta)
        self.get_params(beta=beta)
        return (I - self._theta)/self._gamma + 1./erf_times_gauss((-I + self._theta)/np.sqrt(self._gamma))/np.sqrt(self._gamma)

    def mean2_from_inputs(self,I,I0=None,beta=1):
        I = self.get_input(I,I0=I0,beta=beta)
        self.get_params(beta=beta)
        return 1/self._gamma * (1 +     ((I - self._theta)/np.sqrt(self._gamma))**2  -  ((-I + self._theta)/np.sqrt(self._gamma))/erf_times_gauss((-I + self.theta)/np.sqrt(self._gamma)) )

    def mean12_from_inputs(self,I,I0=None,beta=1):
        I = self.get_input(I,I0=I0,beta=beta)
        self.get_params(beta=beta)
        I_plus = (-I + self._theta)/np.sqrt(self._gamma)
        etg_plus = erf_times_gauss(I_plus)
        mean = (-I_plus+1/etg_plus)/np.sqrt(self._gamma)
        mean2 = 1/self._gamma * (1+ I_plus**2 - I_plus/etg_plus)
        return mean,mean2

    def var_from_inputs(self,I,I0=None,beta=1):
        mean,mean2 = self.mean12_from_inputs(I,I0=I0,beta=beta)
        return mean2 - mean**2

    def cgf_from_inputs(self,I,I0=None,beta=1):
        I = self.get_input(I,I0=I0,beta=beta)
        self.get_params(beta=beta)
        return log_erf_times_gauss((-I + self._theta)/np.sqrt(self._gamma) )  -0.5 * np.log(self._gamma)

    def transform(self,I):
        self.get_params(beta=1)
        return np.maximum(I - self._theta,0)/self._gamma

    def sample_from_inputs(self,I,I0=None,beta=1,out = None,**kwargs):
        I = self.get_input(I,I0=I0,beta=beta)
        self.get_params(beta=beta)
        I_plus = (-I + self._theta)/np.sqrt(self._gamma)
        etg_plus = erf_times_gauss(I_plus)
        rmin = erf( I_plus/np.sqrt(2) )
        rmax = 1
        tmp = (rmax - rmin > 1e-14)
        h =  (np.sqrt(2) * erfinv(rmin + (rmax - rmin) * self.random_state.random_sample(size = I.shape).astype(curr_float)   ) - I_plus )/np.sqrt(self._gamma)
        h[np.isinf(h) | np.isnan(h)] = 0
        return h

    def energy(self,config,remove_init = False,beta=1):
        if remove_init:
            return np.dot(config**2, self.gamma - self.gamma0)/2 + np.dot(config, self.theta - self.theta0)
        else:
            self.get_params(beta=beta)
            return (config**2 * self._gamma).sum(-1)/2 + (config * self._theta).sum(-1)

    def internal_gradients(self,data_pos,data_neg,weights = None,weights_neg=None,value='data',value_neg=None,**kwargs):
        gradients = {}
        if value_neg is None:
            value_neg = value
        if value == 'data':
            mu2_pos = average(data_pos**2,weights=weights)
            mu_pos = average(data_pos,weights=weights)
        elif value == 'mean':
            print('ReLU mean not supported for internal gradient')
        elif value == 'input':
            mu2_pos = average(self.mean2_from_inputs(data_pos),weights=weights)
            mu_pos = average(self.mean_from_inputs(data_pos),weights=weights)
        if value_neg == 'data':
            mu2_neg = average(data_neg**2,weights=weights_neg)
            mu_neg = average(data_neg,weights=weights_neg)
        elif value_neg == 'mean':
            print('ReLU mean not supported for internal gradient')
        elif value_neg == 'input':
            mu2_neg = average(self.mean2_from_inputs(data_neg),weights=weights_neg)
            mu_neg = average(self.mean_from_inputs(data_neg),weights=weights_neg)

        gradients['gamma'] = -0.5 * (mu2_pos - mu2_neg)
        gradients['theta'] = -mu_pos+mu_neg

        if weights is not None:
            gradients['gamma'] *= weights.mean()
            gradients['theta'] *= weights.mean()
        return gradients

    def init_params_from_data(self,X,eps=1e-6,mean=False,weights=None):
        if X is None:
            self.gamma = np.ones(self.N,dtype=curr_float)
            self.gamma0 = np.ones(self.N,dtype=curr_float)
            self.theta = np.zeros(self.N,dtype=curr_float)
            self.theta0 = np.zeros(self.N,dtype=curr_float)
        else:
            mu = average(X,weights=weights)
            var = average(X**2,weights=weights) - mu**2
            self.gamma = 1/(var+eps)
            self.theta = - self.gamma * mu
            self.gamma0 = self.gamma.copy()
            self.theta0 = self.theta.copy()

    def batch_norm_update(self,mu_I,I,lr=1,weights=None):
        delta_mu_I = (mu_I-self.mu_I)
        self.mu_I = mu_I
        self.theta += delta_mu_I
        e = self.mean_from_inputs(I) * self.gamma[np.newaxis,:]
        v = (self.var_from_inputs(psi_pos) * self.gamma[np.newaxis,:]-1)
        var_e = average(e**2,weights=weights) - average(e,weights=weights)**2
        mean_v = average(v,weights=weights)
        new_gamma = (1+mean_v+np.sqrt( (1+mean_v)**2+4*var_e))/2
        self.gamma = np.maximum( self.gamma_min,
        np.maximum(
        (1-lr) * self.gamma + lr * new_gamma,
        self.gamma_drop_max * self.gamma
        )
        )

    def batch_norm_update_gradient(self,gradient_W, gradient_hlayer,V,I,mu,n_c,weights=None):
        dtheta_dw,dgamma_dtheta, dgamma_dw = batch_norm_utils.get_cross_derivatives_ReLU_plus(V,I, self, n_c,weights=weights)
        add_to_gradient(gradient_hlayer['theta'], gradient_hlayer['gamma'], dgamma_dtheta)
        add_to_gradient(gradient_W, gradient_hlayer['theta'], dtheta_dw)
        add_to_gradient(gradient_W, gradient_hlayer['gamma'], dgamma_dw)
        return


class ReLULayer(Layer):
    def __init__(self,N=100,position='visible', batch_norm=False, random_state = None,**kwargs):
        super(ReLULayer, self).__init__(N = N, nature='ReLU',position=position, batch_norm=batch_norm, n_c=1,random_state=random_state)
        self.gamma = np.ones(self.N,dtype=curr_float)
        self.gamma0 = np.ones(self.N,dtype=curr_float)
        self.theta_plus = np.zeros(self.N,dtype=curr_float)
        self.theta_plus0 = np.zeros(self.N,dtype=curr_float)
        self.theta_minus = np.zeros(self.N,dtype=curr_float)
        self.theta_minus0 = np.zeros(self.N,dtype=curr_float)

        self.theta = np.zeros(self.N,dtype=curr_float) # batch norm parametrization.
        self.delta = np.zeros(self.N,dtype=curr_float) # batch norm parametrization.

        self.list_params = ['gamma','theta_plus','theta_minus']
        self.params_anneal = {'gamma':True,'theta_plus':True,'theta_minus':True}
        self.params_newaxis = {'gamma':True,'theta_plus':True,'theta_minus':True}
        if self.position == 'visible':
            self.do_grad_updates = {'gamma':True,'theta_plus':True,'theta_minus':True,
            'theta':False,'delta':False,
            'gamma0':False,'theta_plus0':False,'theta_minus0':False}
        else:
            self.do_grad_updates = {'gamma':False,'theta_plus':True,'theta_minus':True,
            'theta':False,'delta':False,
             'gamma0':True,'theta_plus0':True,'theta_minus0':True}

        self.do_grad_updates_batch_norm = {'gamma':False,'theta':True,'delta':True,
        'theta_plus':False,'theta_minus':False, 'gamma0':False,'theta_plus0':False,'theta_minus0':False}


    def mean_from_inputs(self,I,I0=None,beta=1):
        I = self.get_input(I,I0=I0,beta=beta)
        self.get_params(beta=beta)
        etg_plus = erf_times_gauss((-I + self._theta_plus)/np.sqrt(self._gamma))
        etg_minus = erf_times_gauss((I + self._theta_minus)/np.sqrt(self._gamma))
        p_plus = 1/(1+ etg_minus/etg_plus)
        p_minus = 1- p_plus
        mean_neg = (I + self._theta_minus)/self._gamma - 1/etg_minus/np.sqrt(self._gamma)
        mean_pos = (I - self._theta_plus)/self._gamma + 1/etg_plus/np.sqrt(self._gamma)
        return mean_pos * p_plus + mean_neg * p_minus

    def mean2_from_inputs(self,I,I0=None,beta=1):
        I = self.get_input(I,I0=I0,beta=beta)
        self.get_params(beta=beta)
        etg_plus = erf_times_gauss((-I + self._theta_plus)/np.sqrt(self._gamma))
        etg_minus = erf_times_gauss((I + self._theta_minus)/np.sqrt(self._gamma))
        p_plus = 1/(1+ etg_minus/etg_plus)
        p_minus = 1- p_plus
        mean2_pos = 1/self._gamma * (1 +     ((I - self._theta_plus)/np.sqrt(self._gamma))**2  -  ((-I + self._theta_plus)/np.sqrt(self._gamma))/etg_plus )
        mean2_neg = 1/self._gamma * (1 +     ((I + self._theta_minus)/np.sqrt(self._gamma))**2  -  ((I + self._theta_minus)/np.sqrt(self._gamma))/etg_minus)
        return (p_plus * mean2_pos + p_minus * mean2_neg)

    def mean_pm_from_inputs(self,I,I0=None,beta=1):
        I = self.get_input(I,I0=I0,beta=beta)
        self.get_params(beta=beta)
        etg_plus = erf_times_gauss((-I + self._theta_plus)/np.sqrt(self._gamma))
        etg_minus = erf_times_gauss((I + self._theta_minus)/np.sqrt(self._gamma))
        p_plus = 1/(1+ etg_minus/etg_plus)
        p_minus = 1- p_plus
        mean_neg = (I + self._theta_minus)/self._gamma - 1/etg_minus/np.sqrt(self._gamma)
        mean_pos = (I - self._theta_plus)/self._gamma + 1/etg_plus/np.sqrt(self._gamma)
        return (mean_pos * p_plus, mean_neg * p_minus)

    def mean12_pm_from_inputs(self,I,I0=None,beta=1):
        I = self.get_input(I,I0=I0,beta=beta)
        self.get_params(beta=beta)
        etg_plus = erf_times_gauss((-I + self._theta_plus)/np.sqrt(self._gamma))
        etg_minus = erf_times_gauss((I + self._theta_minus)/np.sqrt(self._gamma))
        p_plus = 1/(1+ etg_minus/etg_plus)
        p_minus = 1- p_plus
        mean_neg = (I + self._theta_minus)/self._gamma - 1/etg_minus/np.sqrt(self._gamma)
        mean_pos = (I - self._theta_plus)/self._gamma + 1/etg_plus/np.sqrt(self._gamma)
        mean2_pos = 1/self._gamma * (1 +     ((I - self._theta_plus)/np.sqrt(self._gamma))**2  -  ((-I + self._theta_plus)/np.sqrt(self._gamma))/etg_plus )
        mean2_neg = 1/self._gamma * (1 +     ((I + self._theta_minus)/np.sqrt(self._gamma))**2  -  ((I + self._theta_minus)/np.sqrt(self._gamma))/etg_minus)
        return (mean_pos * p_plus, mean_neg * p_minus,mean2_pos * p_plus, mean2_neg * p_minus)

    def var_from_inputs(self,I,I0=None,beta=1):
        (mu_pos, mu_neg,mu2_pos,mu2_neg) = self.mean12_pm_from_inputs(I,I0=I0,beta=beta)
        return (mu2_pos + mu2_neg) - (mu_pos + mu_neg)**2

    def cgf_from_inputs(self,I,I0=None,beta=1):
        I = self.get_input(I,I0=I0,beta=beta)
        self.get_params(beta=beta)
        return np.logaddexp( log_erf_times_gauss((-I + self._theta_plus)/np.sqrt(self._gamma) ), log_erf_times_gauss ( (I + self._theta_minus)/np.sqrt(self._gamma))) - 0.5 * np.log(self._gamma)

    def transform(self,I):
        self.get_params(beta=1)
        return 1/self._gamma * ( (I+self._theta_minus) * (I <= np.minimum(-self._theta_minus,(self._theta_plus-self._theta_minus)/2 )) + (I-self._theta_plus) * (I>= np.maximum(self._theta_plus,(self._theta_plus-self._theta_minus)/2 ) ) )

    def sample_from_inputs(self,I,I0=None,beta=1,out=None,**kwargs):
        I = self.get_input(I,I0=I0,beta=beta)
        self.get_params(beta=beta)
        I_plus = (-I + self._theta_plus)/np.sqrt(self._gamma)
        I_minus = (I + self._theta_minus)/np.sqrt(self._gamma)

        etg_plus = erf_times_gauss(I_plus)
        etg_minus = erf_times_gauss(I_minus)

        p_plus = 1/(1+ etg_minus/etg_plus )
        nans  = np.isnan(p_plus)
        p_plus[nans] = 1.0 * (np.abs(I_plus[nans]) > np.abs(I_minus[nans]) )
        p_minus = 1- p_plus

        is_pos = self.random_state.random_sample(size=I.shape) < p_plus
        rmax = np.zeros(I.shape,dtype=curr_float)
        rmin = np.zeros(I.shape,dtype=curr_float)
        rmin[is_pos] = erf( I_plus[is_pos]/np.sqrt(2) )
        rmax[is_pos] = 1
        rmin[~is_pos] = -1
        rmax[~is_pos] = erf( -I_minus[~is_pos]/np.sqrt(2) )

        h = np.zeros(I.shape,dtype=curr_float)
        tmp = (rmax - rmin > 1e-14)
        h = np.sqrt(2) * erfinv(rmin + (rmax - rmin) * self.random_state.random_sample(size = h.shape).astype(curr_float)   )
        h[is_pos] -= I_plus[is_pos]
        h[~is_pos] += I_minus[~is_pos]
        h/= np.sqrt(self._gamma)
        h[np.isinf(h) | np.isnan(h) | ~tmp] = 0
        return h

    def energy(self,config,remove_init = False,beta=1):
        if remove_init:
            return np.dot(config**2, self.gamma - self.gamma0)/2 + np.dot(np.maximum(config,0), self.theta_plus - self.theta_plus0) + np.dot(np.maximum(-config,0), self.theta_minus - self.theta_minus0)
        else:
            self.get_params(beta=beta)
            return (config**2 * self._gamma).sum(-1)/2 + (np.maximum(config,0) * self._theta_plus).sum(-1) + (np.maximum(-config,0) * self._theta_minus).sum(-1)

    def internal_gradients(self,data_pos,data_neg,weights = None,weights_neg=None,value='data',value_neg=None,**kwargs):
        gradients = {}
        if value_neg is None:
            value_neg = value
        if value == 'data':
            mu2_pos = average(data_pos**2,weights=weights)
            mu_p_pos = average(np.maximum(data_pos,0),weights=weights)
            mu_n_pos = average(np.minimum(data_pos,0),weights=weights)
        elif value == 'mean':
            print('ReLU mean not supported for internal gradient')
        elif value == 'input':
            mu2_pos = average(self.mean2_from_inputs(data_pos),weights=weights)
            mu_p_pos,mu_n_pos = self.mean_pm_from_inputs(data_pos)
            mu_p_pos = average(mu_p_pos,weights = weights)
            mu_n_pos = average(mu_n_pos,weights = weights)

        if value_neg == 'data':
            mu2_neg = average(data_neg**2,weights=weights_neg)
            mu_p_neg = average(np.maximum(data_neg,0),weights=weights_neg)
            mu_n_neg = average(np.minimum(data_neg,0),weights=weights_neg)
        elif value_neg == 'mean':
            print('ReLU mean not supported for internal gradient')
        elif value_neg == 'input':
            mu2_neg = average(self.mean2_from_inputs(data_neg),weights=weights_neg)
            mu_p_neg,mu_n_neg = self.mean_pm_from_inputs(data_neg)
            mu_p_neg = average(mu_p_neg,weights=weights_neg)
            mu_n_neg = average(mu_n_neg,weights=weights_neg)

        gradients['gamma'] = -0.5 * (mu2_pos - mu2_neg)
        gradients['theta_plus'] = - mu_p_pos + mu_p_neg
        gradients['theta_minus'] = mu_n_pos - mu_n_neg

        if weights is not None:
            gradients['gamma'] *= weights.mean()
            gradients['theta_plus'] *= weights.mean()
            gradients['theta_minus'] *= weights.mean()

        if self.batch_norm:
            gradients['theta'] = gradients['theta_plus'] - gradients['theta_minus']
            gradients['delta'] = gradients['theta_plus'] + gradients['theta_minus']
        return gradients

    def init_params_from_data(self,X,eps=1e-6,mean=False,weights=None):
        if X is None:
            self.gamma = np.ones(self.N,dtype=curr_float)
            self.gamma0 = np.ones(self.N,dtype=curr_float)
            self.theta_plus = np.zeros(self.N,dtype=curr_float)
            self.theta_plus0 = np.zeros(self.N,dtype=curr_float)
            self.theta_minus = np.zeros(self.N,dtype=curr_float)
            self.theta_minus0 = np.zeros(self.N,dtype=curr_float)
            self.theta = np.zeros(self.N,dtype=curr_float) # batch norm parametrization.
            self.delta = np.zeros(self.N,dtype=curr_float) # batch norm parametrization.
        else:
            mu = average(X,weights=weights)
            var = average(X**2,weights=weights) - mu**2
            self.gamma = 1/(var+eps)
            self.theta_plus = - self.gamma * mu
            self.theta_minus = - self.theta_plus
            self.gamma0 = self.gamma.copy()
            self.theta_plus0 = self.theta_plus.copy()
            self.theta_minus0 = self.theta_minus.copy()

    def batch_norm_update(self,mu_I,I,lr=1,weights=None):
        delta_mu_I = (mu_I-self.mu_I)
        self.mu_I = mu_I
        self.theta += delta_mu_I
        self.theta_plus += delta_mu_I
        self.theta_minus -= delta_mu_I
        e = self.mean_from_inputs(I) * self.gamma[np.newaxis,:]
        v = (self.var_from_inputs(I) * self.gamma[np.newaxis,:]-1)
        var_e = average(e**2,weights=weights) - average(e,weights=weights)**2
        mean_v = average(v,weights=weights)
        new_gamma = (1+mean_v+np.sqrt( (1+mean_v)**2+4*var_e))/2
        self.gamma = np.maximum( self.gamma_min,
        np.maximum(
        (1-lr) * self.gamma + lr * new_gamma,
        self.gamma_drop_max * self.gamma
        )
        )

    def batch_norm_update_gradient(self,gradient_W, gradient_hlayer,V,I,mu,n_c,weights=None):
        dtheta_dw,dgamma_dtheta, dgamma_ddelta, dgamma_dw = batch_norm_utils.get_cross_derivatives_ReLU(V,I, self, n_c,weights=weights)
        add_to_gradient(gradient_hlayer['theta'], gradient_hlayer['gamma'], dgamma_dtheta)
        add_to_gradient(gradient_hlayer['delta'], gradient_hlayer['delta'], dgamma_ddelta)
        add_to_gradient(gradient_W, gradient_hlayer['theta'], dtheta_dw)
        add_to_gradient(gradient_W, gradient_hlayer['gamma'], dgamma_dw)
        return

    def recompute_params(self,which='regular'):
        if which == 'regular':
            self.theta_plus = self.delta + self.theta
            self.theta_minus = self.delta - self.theta
        else:
            self.delta = (self.theta_plus + self.theta_minus)/2
            self.theta = (self.theta_plus - self.theta_minus)/2




class dReLULayer(Layer):
    def __init__(self,N=100,position='visible', batch_norm=False, random_state = None,**kwargs):
        super(dReLULayer, self).__init__(N = N, nature='dReLU',position=position, batch_norm=batch_norm, n_c=1,random_state=random_state)
        self.gamma_plus = np.ones(self.N,dtype=curr_float)
        self.gamma_plus0 = np.ones(self.N,dtype=curr_float)
        self.gamma_minus = np.ones(self.N,dtype=curr_float)
        self.gamma_minus0 = np.ones(self.N,dtype=curr_float)
        self.theta_plus = np.zeros(self.N,dtype=curr_float)
        self.theta_plus0 = np.zeros(self.N,dtype=curr_float)
        self.theta_minus = np.zeros(self.N,dtype=curr_float)
        self.theta_minus0 = np.zeros(self.N,dtype=curr_float)

        self.gamma = np.ones(self.N,dtype=curr_float) # batch norm parametrization.
        self.theta = np.zeros(self.N,dtype=curr_float) # batch norm parametrization.
        self.delta = np.zeros(self.N,dtype=curr_float) # batch norm parametrization.
        self.eta = np.zeros(self.N,dtype=curr_float) # batch norm parametrization.

        self.list_params = ['gamma_plus','gamma_minus','theta_plus','theta_minus']
        self.params_anneal = {'gamma_plus':True,'gamma_minus':True,'theta_plus':True,'theta_minus':True}
        self.params_newaxis = {'gamma_plus':True,'gamma_minus':True,'theta_plus':True,'theta_minus':True}

        if self.position == 'visible':
            self.do_grad_updates = {'gamma_plus':True,'gamma_minus':True,'theta_plus':True,'theta_minus':True,
            'theta':False,'delta':False,'eta':False,'gamma':False,
            'gamma_plus0':False,'gamma_minus0':False,'theta_plus0':False,'theta_minus0':False}
        else:
            self.do_grad_updates = {'gamma_plus':False,'gamma_minus':False,'theta_plus':False,'theta_minus':False,
            'theta':True,'delta':True,'eta':True,'gamma':False,
             'gamma_plus0':True,'gamma_minus0':True,'theta_plus0':True,'theta_minus0':True}

        self.do_grad_updates_batch_norm = {'gamma_plus':False,'gamma_minus':False,'theta_plus':False,'theta_minus':False,
        'theta':True,'delta':True,'eta':True,'gamma':False,
         'gamma_plus0':True,'gamma_minus0':True,'theta_plus0':True,'theta_minus0':True}

    def mean_from_inputs(self,I,I0=None,beta=1):
        I = self.get_input(I,I0=I0,beta=beta)
        self.get_params(beta=beta)
        I_plus = (-I + self._theta_plus)/np.sqrt(self._gamma_plus)
        I_minus = (I + self._theta_minus)/np.sqrt(self._gamma_minus)

        etg_plus = erf_times_gauss(I_plus)
        etg_minus = erf_times_gauss(I_minus)

        p_plus = 1/(1+ (etg_minus/np.sqrt(self._gamma_minus))/(etg_plus/np.sqrt(self._gamma_plus)) )
        nans  = np.isnan(p_plus)
        p_plus[nans] = 1.0 * (np.abs(I_plus[nans]) > np.abs(I_minus[nans]) )
        p_minus = 1- p_plus
        mean_pos = (-I_plus + 1/etg_plus) / np.sqrt(self._gamma_plus)
        mean_neg = (I_minus - 1/etg_minus) / np.sqrt(self._gamma_minus)
        return mean_pos * p_plus + mean_neg * p_minus

    def mean2_from_inputs(self,I,I0=None,beta=1):
        I = self.get_input(I,I0=I0,beta=beta)
        self.get_params(beta=beta)
        I_plus = (-I + self._theta_plus)/np.sqrt(self._gamma_plus)
        I_minus = (I + self._theta_minus)/np.sqrt(self._gamma_minus)

        etg_plus = erf_times_gauss(I_plus)
        etg_minus = erf_times_gauss(I_minus)

        p_plus = 1/(1+ (etg_minus/np.sqrt(self._gamma_minus))/(etg_plus/np.sqrt(self._gamma_plus)) )
        nans  = np.isnan(p_plus)
        p_plus[nans] = 1.0 * (np.abs(I_plus[nans]) > np.abs(I_minus[nans]) )
        p_minus = 1- p_plus
        mean2_pos = 1/self._gamma_plus * (1 +     I_plus**2  -  I_plus/etg_plus )
        mean2_neg = 1/self._gamma_minus * (1 +     I_minus**2  -  I_minus/etg_minus )
        return mean2_pos * p_plus + mean2_neg * p_minus

    def mean12_pm_from_inputs(self,I,I0=None,beta=1):
        I = self.get_input(I,I0=I0,beta=beta)
        self.get_params(beta=beta)
        I_plus = (-I + self._theta_plus)/np.sqrt(self._gamma_plus)
        I_minus = (I + self._theta_minus)/np.sqrt(self._gamma_minus)

        etg_plus = erf_times_gauss(I_plus)
        etg_minus = erf_times_gauss(I_minus)

        p_plus = 1/(1+ (etg_minus/np.sqrt(self._gamma_minus))/(etg_plus/np.sqrt(self._gamma_plus)) )
        nans  = np.isnan(p_plus)
        p_plus[nans] = 1.0 * (np.abs(I_plus[nans]) > np.abs(I_minus[nans]) )
        p_minus = 1- p_plus
        mean_pos = (-I_plus + 1/etg_plus) / np.sqrt(self._gamma_plus)
        mean_neg = (I_minus - 1/etg_minus) / np.sqrt(self._gamma_minus)
        mean2_pos = 1/self._gamma_plus * (1 +     I_plus**2  -  I_plus/etg_plus )
        mean2_neg = 1/self._gamma_minus * (1 +     I_minus**2  -  I_minus/etg_minus )
        return (p_plus* mean_pos, p_minus * mean_neg, p_plus * mean2_pos, p_minus * mean2_neg)

    def var_from_inputs(self,I,I0=None,beta=1):
        (mu_pos, mu_neg,mu2_pos,mu2_neg) = self.mean12_pm_from_inputs(I,I0=I0,beta=beta)
        return (mu2_pos + mu2_neg) - (mu_pos + mu_neg)**2

    def cgf_from_inputs(self,I,I0=None,beta=1):
        I = self.get_input(I,I0=I0,beta=beta)
        self.get_params(beta=beta)
        return np.logaddexp( log_erf_times_gauss((-I + self._theta_plus)/np.sqrt(self._gamma_plus) ) - 0.5* np.log(self._gamma_plus), log_erf_times_gauss ( (I + self._theta_minus)/np.sqrt(self._gamma_minus))- 0.5* np.log(self._gamma_minus))

    def transform(self,I):
        self.get_params(beta=1)
        return ( (I+self._theta_minus) * (I <= np.minimum(-self._theta_minus,(self._theta_plus/np.sqrt(self._gamma_plus)-self._theta_minus/np.sqrt(self._gamma_minus) )/(1/np.sqrt(self._gamma_plus) + 1/np.sqrt(self._gamma_minus) ) ) ))/self._gamma_minus \
    + ((I-self._theta_plus) * (I>= np.maximum(self._theta_plus,(self._theta_plus/np.sqrt(self._gamma_plus)-self._theta_minus/np.sqrt(self._gamma_minus) )/(1/np.sqrt(self._gamma_plus) + 1/np.sqrt(self._gamma_minus) ) )))/self._gamma_plus


    def sample_from_inputs(self,I,I0=None,beta=1,out=None,**kwargs):
        I = self.get_input(I,I0=I0,beta=beta)
        self.get_params(beta=beta)
        if out is None:
            out = np.empty_like(I)
        if type(beta) == np.ndarray:
            cy_utilities.sample_from_inputs_dReLU_numba3(I,self._gamma_plus[:,0,:],
                                              self._gamma_minus[:,0,:],
                                              self._theta_plus[:,0,:],
                                              self._theta_minus[:,0,:], out )
        else:
            cy_utilities.sample_from_inputs_dReLU_numba2(I,self._gamma_plus[0],
                                  self._gamma_minus[0],
                                  self._theta_plus[0],
                                  self._theta_minus[0],out)
        return out



    def energy(self,config,remove_init = False,beta=1):
        config_plus = np.maximum(config,0)
        config_minus = np.maximum(-config,0)
        if remove_init:
            return np.dot(config_plus**2, self.gamma_plus - self.gamma_plus0)/2. + np.dot(config_minus**2, self.gamma_minus - self.gamma_minus0)/2. + np.dot(config_plus, self.theta_plus - self.theta_plus0) + np.dot(config_minus, self.theta_minus - self.theta_minus0)
        else:
            self.get_params(beta=beta)
            return (config_plus**2 *  self._gamma_plus).sum(-1)/2. + (config_minus**2 * self._gamma_minus).sum(-1)/2. + (config_plus * self._theta_plus).sum(-1) + (config_minus * self._theta_minus).sum(-1)

    def internal_gradients(self,data_pos,data_neg,weights = None,weights_neg=None,value='data',value_neg=None,**kwargs):
        gradients = {}
        if value_neg is None:
            value_neg = value
        if value == 'data':
            mu2_p_pos = average(np.maximum(data_pos,0)**2,weights=weights)
            mu2_n_pos = average(np.minimum(data_pos,0)**2,weights=weights)
            mu_p_pos = average(np.maximum(data_pos,0),weights=weights)
            mu_n_pos = average(np.minimum(data_pos,0),weights=weights)

        elif value == 'mean':
            print('dReLU mean not supported for internal gradient')
        elif value == 'input':
            mu_p_pos,mu_n_pos,mu2_p_pos,mu2_n_pos = self.mean12_pm_from_inputs(data_pos)
            mu_p_pos = average(mu_p_pos,weights = weights)
            mu_n_pos = average(mu_n_pos,weights = weights)
            mu2_p_pos = average(mu2_p_pos,weights=weights)
            mu2_n_pos = average(mu2_n_pos,weights=weights)

        if value_neg == 'data':
            mu2_p_neg = average(np.maximum(data_neg,0)**2,weights=weights_neg)
            mu2_n_neg = average(np.minimum(data_neg,0)**2,weights=weights_neg)
            mu_p_neg = average(np.maximum(data_neg,0),weights=weights_neg)
            mu_n_neg = average(np.minimum(data_neg,0),weights=weights_neg)
        elif value_neg == 'mean':
            print('dReLU mean not supported for internal gradient')
        elif value_neg == 'input':
            mu_p_neg,mu_n_neg,mu2_p_neg,mu2_n_neg = self.mean12_pm_from_inputs(data_neg)
            mu_p_neg = average(mu_p_neg,weights=weights_neg)
            mu_n_neg = average(mu_n_neg,weights=weights_neg)
            mu2_p_neg = average(mu2_p_neg,weights=weights_neg)
            mu2_n_neg = average(mu2_n_neg,weights=weights_neg)

        gradients['gamma_plus'] = - 0.5 * (mu2_p_pos - mu2_p_neg)
        gradients['gamma_minus'] = -0.5 * (mu2_n_pos - mu2_n_neg)
        gradients['theta_plus'] = - mu_p_pos + mu_p_neg
        gradients['theta_minus'] = mu_n_pos - mu_n_neg


        if weights is not None:
           gradients['gamma_plus'] *= weights.mean()
           gradients['gamma_minus'] *= weights.mean()
           gradients['theta_plus'] *= weights.mean()
           gradients['theta_minus'] *= weights.mean()

        if self.position == 'hidden':
            gradients['theta'] = gradients['theta_plus'] - gradients['theta_minus']
            gradients['delta'] = gradients['theta_plus'] + gradients['theta_minus']

            gradients['gamma'] = gradients['gamma_plus']/(1+self.eta) + gradients['gamma_minus']/(1-self.eta)
            gradients['theta'] = gradients['theta_plus'] - gradients['theta_minus']
            gradients['delta'] = gradients['theta_plus']/np.sqrt(1+self.eta) + gradients['theta_minus']/np.sqrt(1-self.eta)
            gradients['eta'] = (- self.gamma/(1+self.eta)**2 * gradients['gamma_plus']
                                + self.gamma/(1-self.eta)**2 * gradients['gamma_minus']
                                - self.theta/(2*np.sqrt(1+self.eta)**3) * gradients['theta_plus']
                                + self.theta/(2*np.sqrt(1-self.eta)**3) * gradients['theta_minus'] )
        return gradients

    def init_params_from_data(self,X,eps=1e-6,mean=False,weights=None):
        if X is None:
            self.gamma_plus = np.ones(self.N,dtype=curr_float)
            self.gamma_plus0 = np.ones(self.N,dtype=curr_float)
            self.gamma_minus = np.ones(self.N,dtype=curr_float)
            self.gamma_minus0 = np.ones(self.N,dtype=curr_float)
            self.theta_plus = np.zeros(self.N,dtype=curr_float)
            self.theta_plus0 = np.zeros(self.N,dtype=curr_float)
            self.theta_minus = np.zeros(self.N,dtype=curr_float)
            self.theta_minus0 = np.zeros(self.N,dtype=curr_float)
            self.gamma = np.ones(self.N,dtype=curr_float) # batch norm parametrization.
            self.theta = np.zeros(self.N,dtype=curr_float) # batch norm parametrization.
            self.delta = np.zeros(self.N,dtype=curr_float) # batch norm parametrization.
            self.eta = np.zeros(self.N,dtype=curr_float) # batch norm parametrization.
        else:
            mu = average(X,weights=weights)
            var = average(X**2,weights=weights) - mu**2
            self.gamma_plus = 1/(var+eps)
            self.gamma_minus = self.gamma_plus.copy()
            self.theta_plus = - self.gamma_plus * mu
            self.theta_minus = - self.theta_plus
            self.gamma_plus0 = self.gamma_plus.copy()
            self.gamma_minus0 = self.gamma_minus.copy()
            self.theta_plus0 = self.theta_plus.copy()
            self.theta_minus0 = self.theta_minus.copy()

    def batch_norm_update(self,mu_I,I,lr=1,weights=None):
        delta_mu_I = (mu_I-self.mu_I)
        self.mu_I = mu_I
        self.theta += delta_mu_I
        self.theta_plus += delta_mu_I
        self.theta_minus -= delta_mu_I
        e = self.mean_from_inputs(I) * self.gamma[np.newaxis,:]
        v = (self.var_from_inputs(I) * self.gamma[np.newaxis,:]-1)
        var_e = average(e**2,weights=weights) - average(e,weights=weights)**2
        mean_v = average(v,weights=weights)
        new_gamma = (1+mean_v+np.sqrt( (1+mean_v)**2+4*var_e))/2
        self.gamma = np.maximum( self.gamma_min,
        np.maximum(
        (1-lr) * self.gamma + lr * new_gamma,
        self.gamma_drop_max * self.gamma
        )
        )
        self.gamma_plus = self.gamma/(1+self.eta)
        self.gamma_minus = self.gamma/(1-self.eta)


    def batch_norm_update_gradient(self,gradient_W, gradient_hlayer,V,I,mu,n_c,weights=None):
        B = V.shape[0]
        if weights is None:
            weights = np.ones(B,dtype=curr_float)
        if n_c==1:
            V = np.asarray(V,dtype=curr_float)
            dtheta_dw,dgamma_dtheta, dgamma_ddelta,dgamma_deta, dgamma_dw = cy_utilities.get_cross_derivatives_dReLU_numba(
            V,I,self.gamma, self.theta, self.eta, self.delta,weights)
        else:
            dtheta_dw,dgamma_dtheta, dgamma_ddelta,dgamma_deta, dgamma_dw =  cy_utilities.get_cross_derivatives_dReLU_Potts_numba(
            V,I,self.gamma, self.theta, self.eta, self.delta,weights,n_c)

        add_to_gradient(gradient_hlayer['theta'], gradient_hlayer['gamma'], dgamma_dtheta)
        add_to_gradient(gradient_hlayer['delta'], gradient_hlayer['gamma'], dgamma_ddelta)
        add_to_gradient(gradient_hlayer['eta'], gradient_hlayer['gamma'], dgamma_deta)
        add_to_gradient(gradient_W, gradient_hlayer['theta'], dtheta_dw)
        add_to_gradient(gradient_W, gradient_hlayer['gamma'], dgamma_dw)
        return


    def recompute_params(self,which='regular'):
        if which == 'regular':
            saturate(self.eta,0.95)
            self.gamma_plus = self.gamma/(1+self.eta)
            self.gamma_minus = self.gamma/(1-self.eta)
            self.theta_plus = self.theta + self.delta/np.sqrt(1+self.eta)
            self.theta_minus = -self.theta + self.delta/np.sqrt(1-self.eta)

        else:
            self.gamma = 2./(1./self.gamma_plus + 1./self.gamma_minus)
            self.eta = (self.gamma/self.gamma_plus - self.gamma/self.gamma_minus)/2
            self.delta = (self.theta_plus + self.theta_minus) * (1/np.sqrt(1+self.eta) + 1/np.sqrt(1-self.eta) )/2
            self.theta = self.theta_plus - self.delta/np.sqrt(1+self.eta)

class Bernoulli_coupledLayer(Layer):
    def __init__(self,N=100,position='visible', batch_norm=False, random_state = None,**kwargs):
        super(Bernoulli_coupledLayer, self).__init__(N = N, nature='Bernoulli_coupled',position=position, batch_norm=batch_norm, n_c=1,random_state=random_state)
        self.fields = np.zeros(self.N,dtype=curr_float)
        self.fields0 = np.zeros(self.N,dtype=curr_float) # useful for PT.
        self.couplings = np.zeros([self.N,self.N],dtype=curr_float)
        self.couplings0 = np.zeros([self.N,self.N],dtype=curr_float)

        self.list_params = ['fields','couplings']
        self.params_anneal = {'fields':True,'couplings':True}
        self.params_newaxis = {'fields':False,'couplings':False}

        self.do_grad_updates = {'fields':True,'couplings':True,
        'fields0': (self.position=='hidden'), 'couplings0':False}
        self.do_grad_updates_batch_norm = self.do_grad_updates


    def sample_from_inputs(self,I,I0=None,beta=1,previous=(None,None),**kwargs):
        if I is None:
            if I0 is not None:
                I = (1-beta) * I0
        else:
            I = self.get_input(I,I0=I0,beta=beta)
        (x,fields_eff) = previous
        if x is None:
            B = I.shape[0]
            x = self.random_state.randint(0,high=2,size=[B,self.N])
        else:
            B = x.shape[0]
        if fields_eff is None:
            fields_eff = self.fields[np.newaxis] + self.compute_output(x,self.couplings)

        if I is not None:
            x,fields_eff=cy_utilities.Bernoulli_Gibbs_input_C(x,fields_eff,I, B, self.N,self.fields0,self.couplings,beta, self.random_state.randint(0,high=self.N,size=[B,self.N]), self.random_state.rand(B,self.N) )
        else:
            x,fields_eff=cy_utilities.Bernoulli_Gibbs_free_C(x,fields_eff, B, self.N,self.fields0,self.couplings,beta, self.random_state.randint(0,high=self.N,size=[B,self.N]), self.random_state.rand(B,self.N) )
        return (x,fields_eff)

    def energy(self,config,beta=1,remove_init=False):
        if remove_init:
            fields = self.fields - self.fields0
            couplings = self.couplings
        else:
            if beta==1:
                fields = self.fields
                couplings = self.couplings
            else:
                fields = beta * self.fields + (1-beta) * self.fields0
                couplings = beta * self.couplings

        return - np.dot(config,fields) - 0.5* (np.dot(config , couplings) * config).sum(1)

    def init_params_from_data(self,X,eps=1e-6,mean=False,weights=None):
        if X is None:
            self.fields = np.zeros(self.N,dtype=curr_float)
            self.fields0 = np.zeros(self.N,dtype=curr_float) # useful for PT.
            self.couplings = np.zeros([self.N,self.N],dtype=curr_float)
            self.couplings0 = np.zeros([self.N,self.N],dtype=curr_float)
        else:
            if mean:
                mu = X
            else:
                mu = average(X,weights=weights)
            self.fields = np.log((mu+ eps)/(1-mu + eps))
            self.fields0 = self.fields.copy()
            self.couplings *=0
            self.couplings0 *=0

    def internal_gradients(self,data_pos,data_neg,weights = None,weights_neg=None,value='data',value_neg=None,l2=0,l1=0,**kwargs):
        gradients = {}
        if value_neg is None:
            value_neg = value
        if value =='data':
            mu_pos = average(data_pos,weights=weights)
            comu_pos = average_product(data_pos, data_pos,weights=weights)
        elif value == 'mean':
            mu_pos = data_pos[0]
            comu_pos = data_pos[1]
        elif value == 'input':
            print('not supported')

        if value_neg == 'data':
            mu_neg = average(data_neg,weights=weights_neg)
            comu_neg = average_product(data_neg, data_neg,weights=weights_neg)
        elif value_neg == 'mean':
            mu_neg = average(data_neg,weights=weights_neg)
            comu_neg = average_product(data_neg, data_neg,weights=weights_neg)
        elif value_neg == 'input':
            print('not supported')


        if self.batch_norm:
            gradients['couplings'] = comu_pos - comu_neg - mu_pos[:,np.newaxis] * (mu_pos-mu_neg)[np.newaxis,:] - mu_pos[np.newaxis,:] * (mu_pos-mu_neg)[:,np.newaxis]
            gradients['fields'] = mu_pos-mu_neg - np.dot(gradient['couplings'],mu_pos)
        else:
            gradients['fields'] = mu_pos-mu_neg
            gradients['couplings'] = comu_pos - comu_neg
        if weights is not None:
            gradients['fields'] *= weights.mean()
            gradients['couplings'] *= weights.mean()

        if l2>0:
            gradients['couplings'] -= l2 * self.couplings
        if l1>0:
            gradients['couplings'] -= l1 * np.sign(self.couplings)
        return gradients




class Spin_coupledLayer(Layer):
    def __init__(self,N=100,position='visible', batch_norm=False, random_state = None,**kwargs):
        super(Spin_coupledLayer, self).__init__(N = N, nature='Spin_coupled',position=position, batch_norm=batch_norm, n_c=1,random_state=random_state)
        self.fields = np.zeros(self.N,dtype=curr_float)
        self.fields0 = np.zeros(self.N,dtype=curr_float) # useful for PT.
        self.couplings = np.zeros([self.N,self.N],dtype=curr_float)
        self.couplings0 = np.zeros([self.N,self.N],dtype=curr_float)

        self.list_params = ['fields','couplings']
        self.params_anneal = {'fields':True,'couplings':True}
        self.params_newaxis = {'fields':False,'couplings':False}

        self.do_grad_updates = {'fields':True,'couplings':True,
        'fields0': (self.position=='hidden'), 'couplings0':False}
        self.do_grad_updates_batch_norm = self.do_grad_updates


    def sample_from_inputs(self,I,I0=None,beta=1,previous=(None,None),**kwargs):
        if I is None:
            if I0 is not None:
                I = (1-beta) * I0
        else:
            I = self.get_input(I,I0=I0,beta=beta)
        (x,fields_eff) = previous

        if x is None:
            B = I.shape[0]
            x = 2*self.random_state.randint(0,high=2,size=[B,self.N])-1
        else:
            B = x.shape[0]

        if fields_eff is None:
            fields_eff = self.fields[np.newaxis] + self.compute_output(x,self.couplings)

        if I is not None:
            x,fields_eff=cy_utilities.Spin_Gibbs_input_C(x,fields_eff,I, B, self.N,self.fields0,self.couplings, beta,self.random_state.randint(0,high=self.N,size=[B,self.N]), self.random_state.rand(B,self.N) )
        else:
            x,fields_eff=cy_utilities.Spin_Gibbs_free_C(x,fields_eff, B, self.N,self.fields0,self.couplings, beta,self.random_state.randint(0,high=self.N,size=[B,self.N]), self.random_state.rand(B,self.N) )
        return (x,fields_eff)

    def energy(self,config,beta=1,remove_init=False):
        if remove_init:
            fields = self.fields - self.fields0
            couplings = self.couplings
        else:
            if beta==1:
                fields = self.fields
                couplings = self.couplings
            else:
                fields = beta * self.fields + (1-beta) * self.fields0
                couplings = beta * self.couplings
        return - np.dot(config,fields) - 0.5* (np.dot(config , couplings) * config).sum(1)

    def init_params_from_data(self,X,eps=1e-6,mean=False,weights=None):
        if X is None:
            self.fields = np.zeros(self.N,dtype=curr_float)
            self.fields0 = np.zeros(self.N,dtype=curr_float) # useful for PT.
            self.couplings = np.zeros([self.N,self.N],dtype=curr_float)
            self.couplings0 = np.zeros([self.N,self.N],dtype=curr_float)
        else:
            if mean:
                mu = X
            else:
                mu = average(X,weights=weights)
            self.fields= 0.5*np.log((1+mu + eps)/(1-mu + eps) )
            self.fields0 = self.fields.copy()
            self.couplings *=0
            self.couplings0 *=0

    def internal_gradients(self,data_pos,data_neg,weights = None,weights_neg=None,value='data',value_neg=None,l2=0,l1=0,**kwargs):
        gradients = {}
        if value_neg is None:
            value_neg = value
        if value =='data':
            mu_pos = average(data_pos,weights=weights)
            comu_pos = average_product(data_pos, data_pos,weights=weights)
        elif value == 'mean':
            mu_pos = data_pos[0]
            comu_pos = data_pos[1]
        elif value == 'input':
            print('not supported')

        if value_neg == 'data':
            mu_neg = average(data_neg,weights=weights_neg)
            comu_neg = average_product(data_neg, data_neg,weights=weights_neg)
        elif value_neg == 'mean':
            mu_neg = average(data_neg,weights=weights_neg)
            comu_neg = average_product(data_neg, data_neg,weights=weights_neg)
        elif value_neg == 'input':
            print('not supported')


        if self.batch_norm:
            gradients['couplings'] = comu_pos - comu_neg - mu_pos[:,np.newaxis] * (mu_pos-mu_neg)[np.newaxis,:] - mu_pos[np.newaxis,:] * (mu_pos-mu_neg)[:,np.newaxis]
            gradients['fields'] = mu_pos-mu_neg - np.dot(gradient['couplings'],mu_pos)
        else:
            gradients['fields'] = mu_pos-mu_neg
            gradients['couplings'] = comu_pos - comu_neg
        if weights is not None:
            gradients['fields'] *= weights.mean()
            gradients['couplings'] *= weights.mean()

        if l2>0:
            gradients['couplings'] -= l2 * self.couplings
        if l1>0:
            gradients['couplings'] -= l1 * np.sign(self.couplings)
        return gradients


class Potts_coupledLayer(Layer):
    def __init__(self,N=100,position='visible', batch_norm=False, gauge='zerosum', n_c=2, random_state = None,**kwargs):
        super(Potts_coupledLayer, self).__init__(N = N, nature='Potts_coupled',position=position, batch_norm=batch_norm, n_c=n_c,random_state=random_state)
        self.fields = np.zeros([self.N,self.n_c],dtype=curr_float)
        self.fields0 = np.zeros([self.N,self.n_c],dtype=curr_float)
        self.couplings = np.zeros([self.N,self.N,self.n_c,self.n_c],dtype=curr_float)
        self.couplings0 = np.zeros([self.N,self.N,self.n_c,self.n_c],dtype=curr_float)
        self.gauge = gauge

        self.list_params = ['fields','couplings']
        self.params_anneal = {'fields':True,'couplings':True}
        self.params_newaxis = {'fields':False,'couplings':False}

        self.do_grad_updates = {'fields':True,'couplings':True,
        'fields0': (self.position=='hidden'), 'couplings0':False}
        self.do_grad_updates_batch_norm = self.do_grad_updates


    def sample_from_inputs(self,I,I0=None,beta=1,previous=(None,None),**kwargs):
        if I is None:
            if I0 is not None:
                I = (1-beta) * I0
        else:
            I = self.get_input(I,I0=I0,beta=beta)
        (x,fields_eff) = previous
        if x is None:
            B = I.shape[0]
            x = self.random_state.randint(0,high=self.n_c,size=[B,self.N])
        else:
            B = x.shape[0]
        if fields_eff is None:
            fields_eff = self.fields[np.newaxis] + self.compute_output(x,self.couplings)

        if I is not None:
            x,fields_eff=cy_utilities.Potts_Gibbs_input_C(x,fields_eff,I, B, self.N,self.n_c,self.fields0,self.couplings, beta,self.random_state.randint(0,high=self.N,size=[B,self.N]), self.random_state.rand(B,self.N) )
        else:
            x,fields_eff=cy_utilities.Potts_Gibbs_free_C(x,fields_eff, B, self.N,self.n_c,self.fields0,self.couplings, beta,self.random_state.randint(0,high=self.N,size=[B,self.N]), self.random_state.rand(B,self.N) )
        return (x,fields_eff)


    def energy(self,config,beta=1,remove_init=False):
        if remove_init:
            fields = self.fields - self.fields0
            couplings = self.couplings
        else:
            if beta==1:
                fields = self.fields
                couplings = self.couplings
            else:
                fields = beta * self.fields + (1-beta) * self.fields0
                couplings = beta * self.couplings
        return - cy_utilities.dot_Potts_C(self.N, self.n_c,config, fields) - 0.5 * bilinear_form(couplings, config,config,c1=self.n_c,c2=self.n_c)

    def init_params_from_data(self,X,eps=1e-6,mean=False,weights=None):
        if X is None:
            self.fields = np.zeros([self.N,self.n_c],dtype=curr_float)
            self.fields0 = np.zeros([self.N,self.n_c],dtype=curr_float)
            self.couplings = np.zeros([self.N,self.N,self.n_c,self.n_c],dtype=curr_float)
            self.couplings0 = np.zeros([self.N,self.N,self.n_c,self.n_c],dtype=curr_float)
        else:
            if mean:
                mu = X
            else:
                mu = average(X,weights=weights,c=self.n_c)
            self.fields = invert_softmax(mu,eps=eps, gauge = self.gauge)
            self.fields0 = self.fields.copy()
            self.couplings *=0
            self.couplings0 *=0

    def internal_gradients(self,data_pos,data_neg,weights = None,weights_neg=None,value='data',value_neg=None,l2=0,l1=0,**kwargs):
        gradients = {}
        if value_neg is None:
            value_neg = value
        if value == 'data':
            mu_pos = average(data_pos,c=self.n_c,weights=weights)
            comu_pos = average_product(data_pos, data_pos,weights=weights,c1=self.n_c, c2= self.n_c)
        elif value == 'mean':
            mu_pos = data_pos[0]
            comu_pos = data_pos[1]
        elif value == 'input':
            print('not supported')
        if value_neg == 'data':
            mu_neg = average(data_neg,c=self.n_c,weights=weights_neg)
            comu_neg = average_product(data_neg, data_neg,weights=weights_neg,c1=self.n_c, c2= self.n_c)
        elif value_neg == 'mean':
            mu_neg = average(data_neg,c=self.n_c,weights=weights_neg)
            comu_neg = average_product(data_neg, data_neg,c1=self.n_c, c2= self.n_c,weights=weights_neg)
        elif value_neg == 'input':
            print('not supported')

        if self.batch_norm:
            gradients['couplings'] = comu_pos - comu_neg - mu_pos[:,np.newaxis,:,np.newaxis] * (mu_pos-mu_neg)[np.newaxis,:,np.newaxis,:] - mu_pos[np.newaxis,:,np.newaxis,:] * (mu_pos-mu_neg)[:,np.newaxis,:,np.newaxis]
            gradients['fields'] = mu_pos - mu_neg - np.tensordot(gradients['couplings'], mu_pos,axes=([1,3],[0,1]))
        else:
            gradients['fields'] =  mu_pos - mu_neg
            gradients['couplings'] = comu_pos - comu_neg

        if weights is not None:
            gradients['fields'] *= weights.mean()
            gradients['couplings'] *= weights.mean()

        if l2>0:
            gradients['couplings'] -= l2 * self.couplings
        if l1>0:
            gradients['couplings'] -= l1 * np.sign(self.couplings)
        return gradients




class PottsInterpolateLayer(PottsLayer): # Useful for Augmented Parallel Tempering.
    def __init__(self,N=100,degree = 2, position='hidden', gauge='zerosum', n_c=2, random_state = None,**kwargs):
        super(PottsInterpolateLayer, self).__init__(N = N, nature='PottsInterpolate',position=position, batch_norm=False, n_c=n_c,random_state=random_state)
        self.degree = degree
        self.fields = np.zeros([degree+1,self.N,self.n_c],dtype=curr_float)

    def get_coefficients(self,beta=1):
        return np.array([0*beta+1,beta] + [beta * (1.0-beta)*4 * (k+2.0)/2.0 * ((k+2.0)/(k+1e-10))**(k/2.0) * ( (beta-0.5)*2 )**k for k in range(self.degree-1) ])

    def get_params(self,beta=1):
        beta_is_array = (type(beta) == np.ndarray)
        if not beta_is_array:
            if beta == 1:
                self._fields = (self.fields[0] + self.fields[1])
            elif beta == 0:
                self._fields = self.fields[0]
            else:
                coefficients = self.get_coefficients(beta)
                self._fields = np.tensordot(coefficients,self.fields,(0,0))
        else:
            coefficients = self.get_coefficients(beta)
            self._fields = np.tensordot(coefficients,self.fields,(0,0))
        if beta_is_array:
            self._fields = self._fields[:,np.newaxis]
        else:
            self._fields = self._fields[np.newaxis]

    def update_fields(self,datas,betas,learning_rate):
        if (self.degree<2) | len(betas)<3:
            return
        else:
            if not hasattr(self,'mu_ref'):
                if self.nature in ['Bernoulli','Spin']:
                    self.mu_ref = self.mean_from_inputs(np.zeros([1,self.N],beta=0) )
                elif self.nature == 'Potts':
                    self.mu_ref = self.mean_from_inputs(np.zeros([1,self.N,self.n_c],beta=0) )
            nbetas = len(betas)
            mu = np.array([utilities.average(datas[l],c=self.n_c) for l in range(1,nbetas-1)])
            var = mu * (1-mu) + 1.0/datas.shape[1]
            coefficients = self.get_coefficients(beta)[2:,1:-1]
            P1 = np.linalg.pinv(coefficients,rcond=0.1)
            P2 = np.linalg.pinv(coefficients.T,rcond=0.1)
            grad = coefficients.sum(1)[:,np.newaxis,np.newaxis] * self.mu_ref[np.newaxis,:,:] -np.tensordot(coefficients, mu,axes=(1,0) )
            direction = np.tensordot(P2,np.tensordot(P1,grad,axes=(1,0))/var, axes=(1,0) )
            direction -= direction.mean(-1)[:,:,np.newaxis]
            self.fields[2:] += learning_rate * direction
