import numpy as np
from sklearn.utils import gen_batches

from abc import ABC, abstractmethod

import pickle
import os

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def kl_divergence(p, q):
    """
    The KL-Divergence measures how different are two probability distributions
    over the same variable. 
    """ 
    assert len(p) == len(q)
    q[q == 0] = 0.00000001 # Not let any element of q be 0
    return sum(p[i] * np.log2(p[i] / q[i]) for i in range(len(p)) if p[i] > 0)


class RBM(ABC):
    """
        Implementation of Restricted Boltzman Machine for
        KISA Optimization Techniques by Sandra Burgos and
        Alan Garcia.
        
        References:
            - https://es.wikipedia.org/wiki/M%C3%A1quina_de_Boltzmann_restringida
            - https://upcommons.upc.edu/bitstream/handle/2117/344139/TFM.pdf?sequence=2&isAllowed=y

        Other resources for implementation:
            1. https://leftasexercise.com/2018/04/09/restricted-boltzmann-machines/
            2. https://leftasexercise.com/2018/04/13/learning-algorithms-for-restricted-boltzmann-machines-contrastive-divergence/
            3. https://leftasexercise.com/2018/04/20/training-restricted-boltzmann-machines-with-persistent-contrastive-divergence/ 
            4. https://christian-igel.github.io/paper/TRBMAI.pdf
    """
    def __init__(self, visible_nodes, hidden_nodes, k: int = 1, pretrained: str = None) -> None:
        """
            n_v: number of visible nodes
            n_h: number of hidden nodes
            k -> Number of gibbsampling steps to be performed    
            pretrained: path of a rbm model checkpoint to load

            W -> Network weights
            a -> Visible layer biases
            b -> Hidden layer biases
        """
        
        if pretrained:
            self._load(pretrained)
        else:
            self.n_v = visible_nodes
            self.n_h = hidden_nodes
            self.k = k 
    
            # Initialize random weights from uniform distribution (Xavier initialization)
            limit = np.sqrt(6. / (self.n_v + self.n_h))
            self.W = np.random.uniform(low=-limit, high=limit, size=(self.n_v, self.n_h))
            self.a = np.random.uniform(low=-limit, high=limit, size=(self.n_v)) # np.zeros((1, visible_nodes))
            self.b = np.random.uniform(low=-limit, high=limit, size=(self.n_h)) # np.zeros((1, hidden_nodes))
    
    def _load(self, filename:str):
        """
            Load the params from a pretrained model 
        """
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()          

        self.__dict__.update(tmp_dict) 

    def save(self, filename:str):
        """
            Save the model params
        """
        base, ext = os.path.splitext(filename)  # Separate name and extension
    
        # Check if the file exists
        while os.path.exists(filename):
            filename = f"{base}_({i}){ext}"
            i += 1

        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def _forward(self, V):
        """ 
            Fordward propagation:
            Activation probability 
                > P(hj = 1 | v) = S (bj + sum_i=1^m(Wi,j * vi) )
                    where S is the sigmoid function, v the visible layers set and h the hidden layers set
                > P(h | v) = S( b + v @ W)
        """
        # V: (N, visible_nodes)
        h_p = np.dot(V, self.W) + self.b
        h_p = sigmoid(h_p)
        assert h_p.shape == (V.shape[0], self.n_h)

        h = np.random.binomial(1, h_p, size=h_p.shape)
        return h_p, h
    
    def _backward(self, H):
        """
            Backward:
            Activation probability > P(vi = 1 | h) = S (ai + sum_j=1^m(Wi,j * hj) )
                where S is the sigmoid function, v the visible layers set and h the hidden layers set
        """
        # V: (N, hidden_nodes)
        v_p = np.dot(H, self.W.T) + self.a
        v_p = sigmoid(v_p)
        assert v_p.shape == (H.shape[0], self.n_v)

        v = np.random.binomial(1, v_p, size=v_p.shape)
        return v_p, v

    def _gibbsampling(self, V):
        """
            Perform Gibbsampling
            OUTPUT: (h_p, h, v_p_, v_)
                h_p: hidden layer activation probs
                h: hidden layer samples
                v_p_: visible layer activation probs
                v_: visible layer reconstruction
        """
        v = V
        for i in range(self.k):
            h_p, h = self._forward(v)
            v_p, v = self._backward(h)
        return h_p, h, v_p, v

    @abstractmethod
    def fit():
        """
        The objective is to minimize the energy E of the system:
            E(v, h)=-a' v - b' h - v' W h
        So, to train the RBM we need to compute these terms:
            ∂Energy(v, h)/∂Wij = −hi · vj
            ∂Energy(v, h)/∂bj = −vj
            ∂Energy(v, h)/∂ci = −hi
        However, we don't have access to v and h but we have access
        to ther probabilities p_v and p_h, so we have to sample from
        P(v | h) and P(h | v)
        But direct sampling from those distributions is hard, thats why 
        CD, PCD and other methods were developed
        """
        raise NotImplemented

    def reconstruct(self, X):
        """
            Gibbsampling step to reconstruct output vk from v0 input.
            OUTPUT: generated visible layer output
        """
        h_p, h, v_p_, v_  = self._gibbsampling(X)
        return v_


class RBM_CD(RBM):
    def __init__(self, visible_nodes, hidden_nodes, k: int = 1) -> None:
       super().__init__(visible_nodes, hidden_nodes, k)
    
    def _contrastive_divergence(self, batch, batch_size, lr):  
        """
            K-Step Contrastive Divergence Algorithm
        """      
        v = batch
        h_p, h, v_p_, v_ = self._gibbsampling(v)
        h_p_, h_ = self._forward(v_)
        
        # v_p_, v_, h_p_ and h_ are on the negative phase
        possitive_association = np.dot(v.T, h_p)
        negative_association = np.dot(v_.T, h_p_)
        
        
        # Update the parameters
        # W += lr * (possitive_associations - negative_associations)
        # possitive_associations = v • p_v
        # negative_associations = vk • p_vk
        self.W += lr * (possitive_association - negative_association) / batch_size
        self.a += lr * np.mean(v - v_, axis=0)
        self.b += lr * np.mean(h_p - h_p_, axis=0)
        
        #Compute loss for the instance
        error = np.sum((v - v_)**2) / batch_size
        return error
   
    def fit(self, X, epochs = 10, batch_dim = 32, lr = 0.01, verbose = False) -> list:
        """
             Train the RBM using Contrastive Divergence
        """
        train_errors = []
        for epoch in range(epochs):
            t_error = 0.0
            train_num = X.shape[0]
            batches = list(gen_batches(train_num, batch_dim))
            for batch in batches:
                batch = X[batch.start:batch.stop]
                batch_size = batch.shape[0]

                error = self._contrastive_divergence(batch, batch_size, lr)
                t_error += error
            
            t_error /= len(batches)
            train_errors.append(t_error)
            if verbose:
                print(f"epoch: {epoch}/{epochs} \t{'error:'} {t_error}")
        
        return train_errors

class RBM_PCD(RBM):
    def __init__(self, visible_nodes, hidden_nodes, batch_size = 32, k: int = 2) -> None:
        super().__init__(visible_nodes, hidden_nodes, k)
        self.current_step = 0 # For learning rate decay

        # Initialize the visible persistent values
        self.visible_persistent = np.random.binomial(1, 0.5, size=(batch_size, self.n_v))  
    
    def _persistent_contrastive_divergence(self, batch, batch_size, lr, weight_decay):
        """
            K-Step Parallel Tempering Algorithm
        """
        
        v = batch
        v_persistent = self.visible_persistent

        h_p, _= self._forward(v)
        _, _, _, v_persistent_ = self._gibbsampling(v_persistent)
        h_p_persistent_, _ = self._forward(v_persistent_)

        possitive_association = np.dot(v.T, h_p)
        negative_association = np.dot(v_persistent_.T, h_p_persistent_)
        
        # Update the parameters
        self.W += lr * (possitive_association - negative_association) / batch_size  - weight_decay * self.W / batch_size
        self.a += lr * np.mean(v - v_persistent_, axis=0)
        self.b += lr * np.mean(h_p - h_p_persistent_, axis=0)
        self.visible_persistent = v_persistent_

        #Compute loss for the instance
        error = np.sum((v - self.visible_persistent)**2) / batch_size
        return error

    def fit(self, X, epochs = 10, batch_dim = 32, lr = 0.1, weight_decay = 0.001, verbose = False) -> list:
        """
            Train the RBM using Persistent Contrastive Divergence
        """
        self.current_step = 0 # lr linear decay

        train_errors = []
        for epoch in range(epochs):
            t_error = 0.0
            train_num = X.shape[0]
            batches = list(gen_batches(train_num, batch_dim))
            for batch in batches:
                batch = X[batch.start:batch.stop]
                batch_size = batch.shape[0]
                
                if batch_size != batch_dim:
                    continue # skip

                # Linear learning rate decay
                learning_rate = lr
                # learning_rate = lr *(1 - self.current_step / (batch_size * len(batches) * epochs))
                # self.current_step += 1

                t_error += self._persistent_contrastive_divergence(batch, batch_size, learning_rate, weight_decay)
           
            t_error /= len(batches)
            train_errors.append(t_error)
            
            if verbose:
                print(f"epoch: {epoch}/{epochs} \t{'error:'} {t_error}")
        return train_errors
