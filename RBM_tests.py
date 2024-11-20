
import numpy as np
from RBM import * 
from datasets import *

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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
        def fit():
            # Train not implemented for just testing 
            # the vissible and hidden layers
            pass
    rbm = RBM_BASE(4, 2)
    p_h, h = rbm._forward(v)

    print("RBM BASE TEST")
    print(f"Test input shape: {v.shape} Test input: \n{v}")
    print(f"rbm visible nodes: {rbm.n_v} hidden nodes: {rbm.n_v}")
    print(f"rbm.a biasses shape: {rbm.a.shape} rbm.a biasses: \n{rbm.a}")
    print(f"rbm.b biasses shape: {rbm.b.shape} rbm.b biasses: \n{rbm.b}")
    print(f"rbm.W weights shape: {rbm.W.shape} rbm.W weights: \n{rbm.W}")
    print(f"rbm forward shape{p_h.shape} rbm forward out: \n{p_h}")

    p_v, v = rbm._backward(p_h)
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
    epochs = 10
    learning_rate = 0.01
    
    # Instantiate the RBM
    rbm = RBM_CD(4, 2, k=10)

    # Test
    print("RBM CD TEST")
    print(f"Test input shape: {v.shape} Test input: \n{v}")
    print(f"Num epochs: {epochs}")
    print(f"rbm visible nodes: {rbm.n_v} hidden nodes: {rbm.n_v}")
    print(f"rbm k-step: {rbm.k} rbm learning rate: {learning_rate}")
    print(f"Initial weights matrix ({rbm.W.shape}): \n{rbm.W}")

    
    rbm.fit(v, epochs=epochs, batch_dim=1, lr=learning_rate)
    
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
    X = np.linspace(-1, 1, num=data_size)
    Y = function(X) # create real ys
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42, stratify=Y)
    train   = np.column_stack(X_train, Y_train)
    test    = np.column_stack(X_test, Y_test)
    
    # Instantiate RBM CD object
    rbm = RBM_CD(X_train.shape[1], hidden_nodes, k = k_gibbs_steps)

    # Test
    print("RBM CD FUNCTION APROXIMATION TEST")
    print(f"Train set size: {train.shape[0]}\tshape: {train.shape}")
    print(f"Test set size:  {test.shape[0]} \tshape: {test.shape }")
    print(f"rbm visible nodes: {rbm.n_v} hidden nodes: {rbm.n_v}")
    print(f"rbm.a biasses shape: {rbm.a.shape} rbm.a biasses: \n{rbm.a}")
    print(f"rbm.b biasses shape: {rbm.b.shape} rbm.b biasses: \n{rbm.b}")
    print(f"rbm.W weights shape: {rbm.W.shape} rbm.W weights: \n{rbm.W}")

    rbm.fit(train, epochs=epochs, batch_dim=1, lr=learning_rate)
    
    
    # Evaluation process
    kls = []
    dev_loss = 0.0
    for v in test:
        v_, _ = rbm.generate(v)
        dev_loss += np.mean(np.abs(v[v >= 0] - v_[v >= 0])) 
        kl = kl_divergence(v.flatten() / np.sum(v.flatten()), v_.flatten() / np.sum(v_.flatten()))
        kls.append(kl)    
        dev_loss /= test.shape[0]
    dev_divergence = sum(kls) / len(kls)
    print(f"Dev loss: {dev_loss},\tDev KL-Divergence: {dev_divergence}")

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
    np.random.seed(42) # For reproducibility
    
    # Hyperparameters
    k_gibbs_steps   = 5
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
    rbm = RBM_CD(xor_logic.shape[1], hidden_nodes, k=k_gibbs_steps)

    # Test
    print("RBM CD FUNCTION APROXIMATION TEST")
    print(f"XOR logic set shape: {xor_logic.shape}\t XOR logic:\n{xor_logic}")
    print(f"rbm visible nodes: {rbm.n_v} hidden nodes: {rbm.n_v}")
    print(f"rbm.a biasses shape: {rbm.a.shape} rbm.a biasses: \n{rbm.a}")
    print(f"rbm.b biasses shape: {rbm.b.shape} rbm.b biasses: \n{rbm.b}")
    print(f"rbm.W weights shape: {rbm.W.shape} rbm.W weights: \n{rbm.W}")

    rbm.fit(xor_logic, epochs=epochs, batch_dim=1, lr=learning_rate)

    print(f"Final rbm.a biasses: \n{rbm.a}")
    print(f"Final rbm.b biasses: \n{rbm.b}")
    print(f"Final rbm.W weights: \n{rbm.W}")


    # Plot the results with matplotlib
    grt_color_map = { 0: "blue", 1: "red" } # Ground_truth color map
    rbm_color_map = { 0: "blue", 1: "red" } # RBM prediction color map

    # Ground truth
    grt_xor_0 = xor_logic[xor_logic[:, 2] == 0]
    grt_xor_1 = xor_logic[xor_logic[:, 2] == 1]
    plt.scatter(grt_xor_0[:, 0], grt_xor_0[:, 1], s = 100, color = grt_color_map[0], label = "grt y = 0", )
    plt.scatter(grt_xor_1[:, 0], grt_xor_1[:, 1], s = 100, color = grt_color_map[1], label = "grt y = 1", )
    
    # Recontruction
    v = np.array([[1, 1, 0]]) # This is not possible for XOR function
    v_ = rbm.reconstruct(v)
    print(f"Impossible inut: {v}") 
    print(f"Reconstructed output: {v_}")
    plt.scatter(v[:, 0], v[:, 1], s = 150, color = "black", label = "Impossible for XOR", alpha=0.6)
    plt.scatter(v_[:, 0], v_[:, 1], s = 150, color = "green", label = "Reconstructed by RBM", alpha=0.6)

    # Show
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("RBM XOR Logic Gate")
    plt.grid()
    plt.legend()
    plt.show()

    print("===================================")
    print()

def test_RBM_CD_MNIST_Reconstruction():
    np.random.seed(42) # For reproducibility
    
    # Hyperparameters
    test_size     = 0.1
    batch_size    = 64
    learning_rate = 0.1
    hidden_nodes  = 30    # Number of visible nodes are the input shape
    epochs        = 100    # Number of epochs
    
    # Load some instances of the dataset
    dataset = MNIST("./data/t10k-images-idx3-ubyte.gz", "./data/t10k-labels-idx1-ubyte.gz")
    X, Y = dataset.to_numpy()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42, stratify=Y)

    # Initialize RBM with 30 hidden units
    rbm = RBM_CD(visible_nodes=X_train.shape[1], hidden_nodes=hidden_nodes)

    # Test
    print("RBM CD MNIST RECONSTRUCTION TEST")
    print(f"Hyperparameters:")
    print(f"\tX_train instances: {X_train.shape[0]}\tX_test instances: {X_test.shape[0]}")
    print(f"\tbatch size: {batch_size}\tepochs: {epochs}\tlearning rate: {learning_rate}")
    print(f"\tvisible nodes: {X_train.shape[1]}\thidden nodes:{hidden_nodes}")
    print(f"Training:")
    rbm.fit(X_train, epochs=epochs, batch_dim=batch_size, lr=learning_rate)
    
    # reconstruction
    digit_indices = [np.where(Y_test == i)[0][0] for i in range(10)]
    resticted_set = X_test[digit_indices]

    reconstructed = rbm.reconstruct(resticted_set)

    # show 10 sample images
    rows = 2
    columns = 10

    fig, axes = plt.subplots(rows, columns,sharey = True,figsize=(30, 6))
    for i in range(rows):
        for j in range(columns):
            if i==0:
              axes[i, j].imshow(resticted_set[j].reshape(28, 28))
            else:
              axes[i, j].imshow(reconstructed[j].reshape(28, 28))

            axes[i, j].tick_params(left = False, right = False , labelleft = False,
                    labelbottom = False, bottom = False)

    axes[0, 0].set_ylabel("ACTUAL", fontsize=12)
    axes[1, 0].set_ylabel("REC.", fontsize=12)
    plt.show()

    print("===================================")
    print()

def test_RBM_PCD_MNIST_Reconstruction():
    np.random.seed(42) # For reproducibility
    
    # Hyperparameters
    test_size     = 0.1
    batch_size    = 64
    learning_rate = 0.1
    hidden_nodes  = 30    # Number of visible nodes are the input shape
    epochs        = 100    # Number of epochs
    
    # Load some instances of the dataset
    dataset = MNIST("./data/t10k-images-idx3-ubyte.gz", "./data/t10k-labels-idx1-ubyte.gz")
    X, Y = dataset.to_numpy()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42, stratify=Y)

    # Initialize RBM with 30 hidden units
    rbm = RBM_PCD(visible_nodes=X_train.shape[1], hidden_nodes=hidden_nodes, batch_size=batch_size, k= 2)

    # Test
    print("RBM PCD MNIST RECONSTRUCTION TEST")
    print(f"Hyperparameters:")
    print(f"\tX_train instances: {X_train.shape[0]}\tX_test instances: {X_test.shape[0]}")
    print(f"\tbatch size: {batch_size}\tepochs: {epochs}\tlearning rate: {learning_rate}")
    print(f"\tvisible nodes: {X_train.shape[1]}\thidden nodes:{hidden_nodes}")
    print(f"Training:")
    rbm.fit(X_train, epochs=epochs, batch_dim=batch_size, lr=learning_rate)

    # reconstruction
    digit_indices = [np.where(Y_test == i)[0][0] for i in range(10)]
    resticted_set = X_test[digit_indices]

    reconstructed = rbm.reconstruct(resticted_set)

    # show 10 sample images
    rows = 2
    columns = 10

    fig, axes = plt.subplots(rows, columns,sharey = True,figsize=(30, 6))
    for i in range(rows):
        for j in range(columns):
            if i==0:
              axes[i, j].imshow(resticted_set[j].reshape(28, 28))
            else:
              axes[i, j].imshow(reconstructed[j].reshape(28, 28))

            axes[i, j].tick_params(left = False, right = False , labelleft = False,
                    labelbottom = False, bottom = False)

    axes[0, 0].set_ylabel("ACTUAL", fontsize=12)
    axes[1, 0].set_ylabel("REC.", fontsize=12)
    plt.show()

    print("===================================")
    print()

if __name__ == "__main__":
    # test_RBM_BASE()
    # test_RBM_CD()
    # # test_RBM_CD_function() # Doesn't work
    # test_RBM_CD_XOR()
    #test_RBM_CD_MNIST_Reconstruction()
    test_RBM_PCD_MNIST_Reconstruction()