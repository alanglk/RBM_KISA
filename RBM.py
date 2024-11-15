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

        # Initialize random weights from uniform distribution
        self.W = np.random.random((visible_nodes, hidden_nodes))
        self.a = np.random.random((1, visible_nodes))
        self.b = np.random.random((1, hidden_nodes))

    def forward(self, v: np.ndarray) -> np.ndarray:
        """ 
        Fordward propagation:
        
        Activation probability 
            > P(hj = 1 | v) = S (bj + sum_i=1^m(Wi,j * vi) )
                where S is the sigmoid function, v the visible layers set and h the hidden layers set
            > P(h | v) = S( b + v @ W)
        """
        aux = self.b + v @ self.W
        p_h = sigmoid(aux)       
        return p_h # Probability of h | v

    def backward(self, h: np.ndarray) -> np.ndarray:
        """
        Backward:
        
        Activation probability > P(vi = 1 | h) = S (ai + sum_j=1^m(Wi,j * hj) )
            where S is the sigmoid function, v the visible layers set and h the hidden layers set
        """
        aux = self.a + h @ self.W.T
        p_v = sigmoid(aux)
        return p_v # Probability of v | h
    
    def generate(self, v0: np.ndarray) -> np.ndarray:
        """
            Gibbsampling to generate output vk from v0 input
        """
        p_h0 = self.forward(v0) # compute hidden activation probs
        for _ in range(self.k):
            hk = np.random.binomial(p=p_h0, n = 1) # sample h 
            p_vk = self.backward(hk)
            vk = np.random.binomial(p=p_vk, n = 1) # sample v
            vk[v0 < 0] = v0[v0 < 0]

        return vk, p_h0


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
        train_size = len(train_set)
        train_loss = 0.0
        # k-step contrastive divergence
        for v0 in train_set:
            vk, p_h0 = self.generate(v0) # Gibbsampling for self.k steps
            p_hk = self.forward(vk)

            # Update the parameters
            # W += lr * (possitive_associations - negative_associations)
            # possitive_associations = v • p_v
            # negative_associations = vk • p_vk
            self.W += self.lr * ( np.dot(v0.T, p_h0) - np.dot(vk.T, p_hk) ) 
            self.a += self.lr * np.sum(v0 - vk)
            self.b += self.lr * np.sum(p_h0 - p_hk)

            # Compute loss for the instance
            # Those that initially where active minus those generated that shoul be active
            train_loss += np.mean(np.abs(v0[v0 >= 0] - vk[v0 >= 0])) 
        return train_loss / train_size


def test_RBM_BASE():
    np.random.seed(42) # For reproducibility
    
    # Example data input
    v = np.array([[1, 0.001, 0.001, 1]])

    # Instantiate the RBM test object
    class RBM_BASE(RBM):
        def __init__(self, visible_nodes, hidden_nodes):
            super().__init__(visible_nodes, hidden_nodes)
        def train():
            # Train not implemented for just testing 
            # the vissible and hidden layers
            pass
    rbm = RBM_BASE(4, 2)
    p_h = rbm.forward(v)

    print("RBM BASE TEST")
    print(f"Test input shape: {v.shape} Test input: \n{v}")
    print(f"rbm visible nodes: {rbm.n_v} hidden nodes: {rbm.n_v}")
    print(f"rbm.a biasses shape: {rbm.a.shape} rbm.a biasses: \n{rbm.a}")
    print(f"rbm.b biasses shape: {rbm.b.shape} rbm.b biasses: \n{rbm.b}")
    print(f"rbm.W weights shape: {rbm.W.shape} rbm.W weights: \n{rbm.W}")
    print(f"rbm forward shape{p_h.shape} rbm forward out: \n{p_h}")

    p_v = rbm.backward(p_h)
    print(f"rbm backward shape{p_v.shape} rbm backward out: \n{p_v}")

    v_, _ = rbm.generate(v)
    print(f"rbm generate process for v input and {rbm.k} gibbsampling steps: \n{v_}")
    
    # The goal is to reduce the divergence between the input probability
    # dsitribution and the output. 
    # v = np.array([1, 0.001, 0.001, 1])
    # v_ = np.array([1, 1, 1, 1])
    # kld = 0.98860 >> 0 -> significant divergence
    kld = kl_divergence(v.flatten() / np.sum(v.flatten()), v_.flatten() / np.sum(v_.flatten()))
    print(f"KL-Divergence between input and final RBM probability output: {kld}")
    print("===================================")
    print()

def test_RBM_CD():
    np.random.seed(42) # For reproducibility
    
    # Example data batch (size 1 instance)
    v = np.array([[1, 0.001, 0.001, 1]])

    # Instantiate the RBM
    rbm = RBM_CD(4, 2, k=10, learning_rate=0.5)

    # Test
    print("RBM CD TEST")
    print(f"Test input shape: {v.shape} Test input: \n{v}")
    print(f"rbm visible nodes: {rbm.n_v} hidden nodes: {rbm.n_v}")
    print(f"rbm k-step: {rbm.k} rbm learning rate: {rbm.lr}")
    print(f"Initial weights matrix ({rbm.W.shape}): \n{rbm.W}")
    
    rbm.train([v]) # Train with just one instance
    
    print(f"Final weights matrix ({rbm.W.shape}): \n{rbm.W}")

    v_, _ = rbm.generate(v)
    print(f"rbm generate process for v after training with {rbm.k} gibbsampling steps: \n{v_}")

    kld = kl_divergence(v.flatten() / np.sum(v.flatten()), v_.flatten() / np.sum(v_.flatten()))
    print(f"KL-Divergence between input and final RBM probability output: {kld}")
    print("===================================")
    print()

def test_RBM_CD_training():
    np.random.seed(42) # For reproducibility
    
    # Hyperparameters
    data_size = 50
    test_size = 0.2

    # Generate synthetic data
    # let's try to model a normal distribution using
    # the implemented Contrastive Divergence RBM algorithm
    x = np.linspace(-10, 10, num=data_size)
    def normal(x,mu,sigma):
        return ( 2.*np.pi*sigma**2. )**-.5 * np.exp( -.5 * (x-mu)**2. / sigma**2. )
    y = normal(x, 0, 1) # sample x from normal distribution

    data = np.vstack((x, y)).T # (x, y)
    indexes = np.random.permutation(data.shape[0]) # mix the elements to generate train and test
    test_count = int(test_size * data.shape[0])
    test = data[indexes[:test_count]]   # Test set
    train = data[indexes[test_count:]]  # Train set
    

    # The objective is to train the RBM with the xs and ys and then
    # just give an x to generate the expected y
    

    pass

if __name__ == "__main__":
    test_RBM_BASE()
    test_RBM_CD()
    test_RBM_CD_training()