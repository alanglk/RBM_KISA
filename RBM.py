import numpy as np

def sigmoid(z):
    return 1/(1 + np.exp(-z))

class RBM:
    """
        Implementation of Restricted Boltzman Machine for
        KISA Optimization Techniques by Sandra Burgos and
        Alan Garcia.
        
        References:
            - https://es.wikipedia.org/wiki/M%C3%A1quina_de_Boltzmann_restringida
            - https://upcommons.upc.edu/bitstream/handle/2117/344139/TFM.pdf?sequence=2&isAllowed=y


        W -> Network weights
        a -> Visible layer biases
        b -> Hidden layer biases
    """
    
    def __init__(self, visible_nodes, hidden_nodes):
        self.n_v = visible_nodes
        self.n_h = hidden_nodes

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

def test():
    np.random.seed(42) # For reproducibility
    
    # Example data input
    v = np.array([[1, 2, 3, 4]])

    # Instantiate the RBM object
    rbm = RBM(4, 2)
    
    p_h = rbm.forward(v)

    print(f"Test input shape: \t{v.shape} Test input: \t{v}")
    print(f"rbm visible nodes: {rbm.n_v} hidden nodes: {rbm.n_v}")
    print(f"rbm.a biasses shape: {rbm.a.shape} rbm.a biasses: \n{rbm.a}")
    print(f"rbm.b biasses shape: {rbm.b.shape} rbm.b biasses: \n{rbm.b}")
    print(f"rbm.W weights shape: {rbm.W.shape} rbm.W weights: \n{rbm.W}")
    print(f"rbm forward shape{p_h.shape} rbm forward out: \n{p_h}")

    p_v = rbm.backward(p_h)
    print(f"rbm backward shape{p_v.shape} rbm backward out: \n{p_v}")


if __name__ == "__main__":
    test()