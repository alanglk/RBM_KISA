import numpy as np
from abc import ABC, abstractmethod

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

        W -> Network weights
        a -> Visible layer biases
        b -> Hidden layer biases
    """
    
    def __init__(self, visible_nodes, hidden_nodes, k = 10):
        self.n_v = visible_nodes
        self.n_h = hidden_nodes

        self.k = k # Number of gibbsampling steps to be performed

        # Initialize random weights from uniform distribution (Xavier initialization)
        limit = np.sqrt(6. / (self.n_v + self.n_h))
        self.W = np.random.uniform(low=-limit, high=limit, size=(self.n_v, self.n_h))
        self.a = np.random.uniform(low=-limit, high=limit, size=(1, self.n_v)) # np.zeros((1, visible_nodes))
        self.b = np.random.uniform(low=-limit, high=limit, size=(1, self.n_h)) # np.zeros((1, hidden_nodes))

    def forward(self, v: np.ndarray) -> np.ndarray:
        """ 
            Fordward propagation:
            
            Activation probability 
                > P(hj = 1 | v) = S (bj + sum_i=1^m(Wi,j * vi) )
                    where S is the sigmoid function, v the visible layers set and h the hidden layers set
                > P(h | v) = S( b + v @ W)
        """
        # Probability of h | v
        p_h = sigmoid(self.b + v @ self.W) 

        # sample H
        h = (np.random.uniform(0, 1, size=p_h.shape) < p_h).astype(np.int32)
        return p_h, h

    def backward(self, h: np.ndarray) -> np.ndarray:
        """
            Backward:

            Activation probability > P(vi = 1 | h) = S (ai + sum_j=1^m(Wi,j * hj) )
                where S is the sigmoid function, v the visible layers set and h the hidden layers set
        """
         
        # Probability of v | h
        p_v = sigmoid(self.a + h @ self.W.T)

        # sample v
        v = (np.random.uniform(0, 1, size=p_v.shape) < p_v).astype(np.int32) 
        return p_v, v
    
    def gibbs_sampling(self, v):  
        """
            Perform Gibbsampling
            OUTPUT:
                (probability, samples) of visible units
        """
        vk = v.copy()
        for i in range(self.k):
            p_h, hk = self.forward(vk)
            p_v, vk = self.backward(hk)
        return vk, p_h
    
    def reconstruct(self, v0: np.ndarray) -> np.ndarray:
        """
            Gibbsampling step to reconstruct output vk from v0 input.
            OUTPUT: generated visible layer output
        """
        p_h, hk = self.forward(v0)
        p_v, vk = self.backward(hk)
        return vk


    @abstractmethod
    def train():
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

class RBM_CD(RBM):
    """
        Contrastive Divergence
    """
    def __init__(self, visible_nodes, hidden_nodes,  k = 10, learning_rate = 0.1):
        super().__init__(visible_nodes, hidden_nodes, k)
        self.lr = learning_rate
    
    def train(self, train_set):
        """
            Train the RBM for one epoch using Contrastive Divergence
        """
        train_size = len(train_set)
        train_loss = 0.0

        # k-step contrastive divergence
        for i in range(train_size):
            v0 = train_set[i]
            v0 = v0.reshape(-1, 1).T
            
            p_h0, h = self.forward(v0)
            positive_associations = np.dot(v0.T, p_h0)

            vk, p_hk = self.gibbs_sampling(v0)
            negative_associations = np.dot(vk.T, p_hk)

            # Update the parameters
            # W += lr * (possitive_associations - negative_associations)
            # possitive_associations = v • p_v
            # negative_associations = vk • p_vk
            dW = positive_associations - negative_associations
            self.W += self.lr * dW / train_size
            self.a += self.lr * np.mean(v0 - vk, axis=0)
            self.b += self.lr * np.mean(p_h0 - p_hk, axis=0)

            # Compute loss for the instance
            # Those that initially where active minus those generated that shoul be active
            # train_loss += np.mean(np.abs(v0[v0 >= 0] - vk[v0 >= 0])) 
            train_loss += np.mean((v0 - vk)**2) # MSE

        return train_loss / train_size
    
