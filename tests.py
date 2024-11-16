
import numpy as np
from RBM import * 
from datasets import *

def test_RBM_BASE():
    np.random.seed(42) # For reproducibility
    
    # Example data input
    v = np.array([
        [0.001, 1, 0.001, 1]
        ])

    # Instantiate the RBM test object
    class RBM_BASE(RBM):
        def __init__(self, visible_nodes, hidden_nodes):
            super().__init__(visible_nodes, hidden_nodes)
        def train():
            # Train not implemented for just testing 
            # the vissible and hidden layers
            pass
    rbm = RBM_BASE(4, 2)
    p_h, h = rbm.forward(v)

    print("RBM BASE TEST")
    print(f"Test input shape: {v.shape} Test input: \n{v}")
    print(f"rbm visible nodes: {rbm.n_v} hidden nodes: {rbm.n_v}")
    print(f"rbm.a biasses shape: {rbm.a.shape} rbm.a biasses: \n{rbm.a}")
    print(f"rbm.b biasses shape: {rbm.b.shape} rbm.b biasses: \n{rbm.b}")
    print(f"rbm.W weights shape: {rbm.W.shape} rbm.W weights: \n{rbm.W}")
    print(f"rbm forward shape{p_h.shape} rbm forward out: \n{p_h}")

    p_v, v = rbm.backward(p_h)
    print(f"rbm backward shape{p_v.shape} rbm backward out: \n{p_v}")

    v_ = rbm.reconstruct(v)
    print(f"rbm reconstruct process for v input and {rbm.k} gibbsampling steps: \n{v_}")
    
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
    v = np.array([[0, 1, 0, 1]])
    epochs = 50

    # Instantiate the RBM
    rbm = RBM_CD(4, 2, k=10, learning_rate=0.01)

    # Test
    print("RBM CD TEST")
    print(f"Test input shape: {v.shape} Test input: \n{v}")
    print(f"Num epochs: {epochs}")
    print(f"rbm visible nodes: {rbm.n_v} hidden nodes: {rbm.n_v}")
    print(f"rbm k-step: {rbm.k} rbm learning rate: {rbm.lr}")
    print(f"Initial weights matrix ({rbm.W.shape}): \n{rbm.W}")

    for e in range(epochs):
        train_loss = rbm.train(v) # Train
        print(f"Epoch {e+1}/{epochs}, Train loss (MSE): {train_loss}")
    
    print(f"Final weights matrix ({rbm.W.shape}): \n{rbm.W}")

    v_ = rbm.reconstruct(v)
    print(f"rbm reconstruct process for v after training with {rbm.k} gibbsampling steps: \n{v_}")

    kld = kl_divergence(v.flatten() / np.sum(v.flatten()), v_.flatten() / np.sum(v_.flatten()))
    print(f"KL-Divergence between input and final RBM probability output: {kld}")
    print("===================================")
    print()

def test_RBM_CD_function():
    """
        Let's try to model a y = x³ function using
        the implemented Contrastive Divergence RBM algorithm.

        The objective is to train the RBM with some xs and ys for
        testing the result by just giving an x to reconstruct the
        expected y_ and compare it with the real y.
    """
    np.random.seed(42) # For reproducibility
    
    # Hyperparameters
    data_size       = 50
    test_size       = 0.2
    k_gibbs_steps   = 10
    learning_rate   = 0.01
    hidden_nodes    = 2     # Number of visible nodes are the input shape
    epochs          = 100    # Number of epochs

    # Generate synthetic data
    def function(x):
        return x ** 3
    x = np.linspace(-1, 1, num=data_size)
    #x_features = np.column_stack([x, np.sin(np.pi * x), np.cos(np.pi * x)]) # Expand dimensions of X
    y = function(x) # create real ys
    data = np.column_stack((x, y))
    data = data.reshape(data_size, 1, data.shape[1]) # (N, 1, d) where N is the number of samples and d the dimension of each sample
    
    train_set, test_set = random_train_test_split(data, test_size=test_size)
    train_set, dev_set = random_train_test_split(data, test_size=test_size)

    # Instantiate RBM CD object
    rbm = RBM_CD(data.shape[2], hidden_nodes, k_gibbs_steps, learning_rate)

    # Test
    print("RBM CD FUNCTION APROXIMATION TEST")
    print(f"Train set size: {len(train_set)}\tshape: {train_set.shape}\tshowing 5 first instances:\n{train_set[:5, :]}")
    print(f"Dev set size:   {len(dev_set)  }\tshape: {dev_set.shape  }\tshowing 5 first instances:\n{dev_set[:5, :]  }")
    print(f"Test set size:  {len(test_set) }\tshape: {test_set.shape }\tshowing 5 first instances:\n{test_set[:5, :] }")
    print(f"rbm visible nodes: {rbm.n_v} hidden nodes: {rbm.n_v}")
    print(f"rbm.a biasses shape: {rbm.a.shape} rbm.a biasses: \n{rbm.a}")
    print(f"rbm.b biasses shape: {rbm.b.shape} rbm.b biasses: \n{rbm.b}")
    print(f"rbm.W weights shape: {rbm.W.shape} rbm.W weights: \n{rbm.W}")

    for e in range(epochs):
        # Train process
        train_loss = rbm.train(train_set)

        # Evaluation process
        kls = []
        dev_loss = 0.0
        for v in dev_set:
            v_, _ = rbm.generate(v)
            dev_loss += np.mean(np.abs(v[v >= 0] - v_[v >= 0])) 
            kl = kl_divergence(v.flatten() / np.sum(v.flatten()), v_.flatten() / np.sum(v_.flatten()))
            kls.append(kl)
            
        dev_loss /= len(dev_set)
        dev_divergence = sum(kls) / len(kls) 

        print(f"Epoch {e+1}/{epochs},\tTrain loss: {train_loss},\tDev loss: {dev_loss},\tDev KL-Divergence: {dev_divergence}")

    print(f"Final rbm.a biasses: \n{rbm.a}")
    print(f"Final rbm.b biasses: \n{rbm.b}")
    print(f"Final rbm.W weights: \n{rbm.W}")

    # Reconstruct y = x³ from x
    reconstructed = np.empty((0, data.shape[2]))
    xs = test_set[:, :, 0].flatten()
    xs.sort()
    for x_val in xs:
        y_dummy = np.random.uniform(low=-1, high=1)
        v = np.array([[x_val, y_dummy]]) # set y = to random -> Unknown
        vk, _ = rbm.generate(v)
        reconstructed = np.vstack((reconstructed, vk))

    print(reconstructed)
    # Show the original and reconstrucion on a figure
    import matplotlib.pyplot as plt
    plt.plot(x, y, color = "blue", label="Groundtruth: y = x³")
    plt.plot(reconstructed[:, 0], reconstructed[:, 1], color="red", label="RBM ys reconstructed")
    plt.legend()
    plt.grid()
    plt.show()

    print("===================================")
    print()
    
def test_RBM_CD_XOR():
    """
        Let's try to model a XOR logic gate using the 
        Contrastive Divergence RBM we have already implemented.

        To make this, we will set the train set as all the posible
        inputs/output of the XOR gate and then we will try to give
        the RBM just some input and check if it is correct or not 
    """

    # Hyperparameters
    k_gibbs_steps   = 100
    learning_rate   = 0.0001
    hidden_nodes    = 2     # Number of visible nodes are the input shape
    epochs          = 10    # Number of epochs

    # Train data
    xor_logic = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 1],
    ])


    # Instantiate the RBM object
    rbm = RBM_CD(xor_logic.shape[2], hidden_nodes, k=k_gibbs_steps, learning_rate=learning_rate)

    # Test
    print("RBM CD FUNCTION APROXIMATION TEST")
    print(f"XOR logic set shape: {xor_logic.shape}\t XOR logic:\n{xor_logic}")
    print(f"rbm visible nodes: {rbm.n_v} hidden nodes: {rbm.n_v}")
    print(f"rbm.a biasses shape: {rbm.a.shape} rbm.a biasses: \n{rbm.a}")
    print(f"rbm.b biasses shape: {rbm.b.shape} rbm.b biasses: \n{rbm.b}")
    print(f"rbm.W weights shape: {rbm.W.shape} rbm.W weights: \n{rbm.W}")

    for e in range(epochs):
        # Train process
        train_loss = rbm.train(xor_logic)
        print(f"Epoch {e+1}/{epochs},\tTrain loss: {train_loss}")

    print(f"Final rbm.a biasses: \n{rbm.a}")
    print(f"Final rbm.b biasses: \n{rbm.b}")
    print(f"Final rbm.W weights: \n{rbm.W}")



    # Plot the results with matplotlib
    import matplotlib.pyplot as plt
    grt_color_map = { 0: "blue", 1: "red" } # Ground_truth color map
    rbm_color_map = { 0: "blue", 1: "red" } # RBM prediction color map

    # Ground truth
    grt_xor_0 = xor_logic[xor_logic[:, :, 2] == 0]
    grt_xor_1 = xor_logic[xor_logic[:, :, 2] == 1]
    plt.scatter(grt_xor_0[:, 0], grt_xor_0[:, 1], s = 100, color = grt_color_map[0], label = "grt y = 0")
    plt.scatter(grt_xor_1[:, 0], grt_xor_1[:, 1], s = 100, color = grt_color_map[1], label = "grt y = 1")
    
    # Recontruction
    v = np.array([[1, 1, 0]]) # This is not possible for XOR function
    v_ = rbm.reconstruct(v)
    print(f"Impossible inut: {v}") 
    print(f"Reconstructed output: {v_}")

    # Show
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("RBM XOR Logic Gate")
    plt.grid()
    plt.legend()
    plt.show()

def test_RBM_CD_MNIST_Reconstruction():
    """
        The idea is to train an RBM with the MNIST dataset
        for comparing the real and reconstructed results of
        some images.
    """
    # Hyperparameters
    data_size       = 1000
    k_gibbs_steps   = 1
    learning_rate   = 0.001
    hidden_nodes    = 30   # Number of visible nodes are the input shape
    epochs          = 30    # Number of epochs

    # Load some instances of the dataset
    dataset = MNIST("./data/t10k-images-idx3-ubyte.gz", "./data/t10k-labels-idx1-ubyte.gz", data_size)
    X, Y = dataset.to_numpy()

    # Instantiate the RBM object
    rbm = RBM_CD(X.shape[1], hidden_nodes, k=k_gibbs_steps, learning_rate=learning_rate)

    # Test
    print("RBM CD MNIST RECONSTRUCTION TEST")
    print(f"MNIST X shape: {X.shape}\t MNIST Y shape:\n{Y.shape}")
    print(f"rbm visible nodes: {rbm.n_v} hidden nodes: {rbm.n_v}")
    print(f"rbm.a biasses shape: {rbm.a.shape}")
    print(f"rbm.b biasses shape: {rbm.b.shape}")
    print(f"rbm.W weights shape: {rbm.W.shape}")

    # train
    for e in range(epochs):
        train_loss = rbm.train(X)
        print(f"Epoch {e+1}/{epochs},\tTrain loss: {train_loss}")   

    # test reconstruction
    import matplotlib.pyplot as plt
    digit_indices = [np.where(Y == i) for i in range(10)]
    f, axes = plt.subplots(2, len(digit_indices), figsize=(10, 2))
    for i, indices in enumerate(digit_indices):
        index = indices[0][0]
        v = X[index]
        y = Y[index]

        v_ = rbm.reconstruct(v)
        axes[0, i].imshow(v.reshape(28, 28))
        axes[0, i].set_title(y)
        axes[1, i].imshow(v_.reshape(28, 28))
        axes[1, i].set_title(y)
    plt.axis('off')
    plt.show()
        

    print("===================================")
    print()



if __name__ == "__main__":
    # test_RBM_BASE()
    # test_RBM_CD()
    # test_RBM_CD_function() # Doesn't work
    # test_RBM_CD_XOR()
    test_RBM_CD_MNIST_Reconstruction()