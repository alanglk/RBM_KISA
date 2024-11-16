import gzip
import numpy as np

def random_train_test_split(X: np.ndarray, Y: np.ndarray, test_size: float = 0.2):
    """
        Generates a train/test split for the data
        INPUT:  
            - data:np.ndarray
                X = [[x1, x2, ...], [x1, x2, ...] ...]
                Y = [y1, y2, y3...]
            - test_size: float
        OUTPUT: (train_set:np.ndarray, test_set:np.ndarray) 
    """
    
    indexes = np.random.permutation(X.shape[0]) # mix the elements to generate train and test
    test_count = int(test_size * X.shape[0])

    x_train = X[indexes[test_count:]]
    y_train = Y[indexes[test_count:]]
    x_test  = X[indexes[:test_count]]
    y_test  = Y[indexes[test_count:]]

    return x_train, y_train, x_test, y_test


class MNIST:
    """
    https://yann.lecun.com/exdb/mnist/
    ubyte format of MNIST images:
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000803(2051) magic number
        0004     32 bit integer  10000            number of images
        0008     32 bit integer  28               number of rows
        0012     32 bit integer  28               number of columns
        0016     unsigned byte   ??               pixel
        0017     unsigned byte   ??               pixel
        ........
        xxxx     unsigned byte   ??               pixel
    
    ubyte format of MNIST labels:
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000801(2049) magic number (MSB first)
        0004     32 bit integer  60000            number of items
        0008     unsigned byte   ??               label
        0009     unsigned byte   ??               label
        ........
        xxxx     unsigned byte   ??               label
    """
    
    def __init__(self, images_path: str, label_path: str, N: int = None) -> None:
        """
            INPUT:
                - images path: path of the dataset images
                - labels path: path of the dataset labels
                - N: if not provided, all the images in the input file are read.
                    Else, N images are read.
        """
        
        self.images_path = images_path
        self.labels_path = label_path

        # Open with gzip to uncompress the files
        self.images_file = gzip.open(self.images_path, "rb")
        self.labels_file = gzip.open(self.labels_path, "rb")
        self.images_file.read(4) # Ignore the magic numbers
        self.labels_file.read(4)

        # Prepare Images data file reader
        self.number_of_items = int.from_bytes(self.images_file.read(4), byteorder="big") # Total number of images
        self.row_num = int.from_bytes(self.images_file.read(4), byteorder="big")    # row pixels of image
        self.col_num = int.from_bytes(self.images_file.read(4), byteorder="big")    # col pixels of image
        
        self.N = self.number_of_items if N is None else N # Number of items to be read
        self.num_bytes_per_image = self.col_num * self.row_num
        self.returned_images = 0    # Number of read images
        self.images_file.seek(16)   # Set the cursor to the init of the first image

        # Prepare Labels file reader
        assert self.number_of_items == int.from_bytes(self.labels_file.read(4), byteorder="big")
        self.labels_file.seek(8)

        # Print metadata
        print(f"[MNIST]     image size: {self.col_num}x{self.row_num} total dataset images: {self.number_of_items} images to be read: {self.N}")

        

    def __iter__(self):
        """Create an iterator for the MNIST dataset"""
        return self

    def __next__(self):
        """Return the next image in the dataset"""
        if self.returned_images >= self.N:
            self.images_file.close()
            self.labels_file.close()
            raise StopIteration
        else:
            pixels = []
            for _ in range(self.num_bytes_per_image):
                data = self.images_file.read(1)

                if not data:
                    print(f"[MNIST]     ERROR -> This should not be reached")
                    self.file.close()
                    raise StopIteration
                
                pixels.append(int.from_bytes(data, byteorder="big"))
            label = int.from_bytes(self.labels_file.read(1))
            self.returned_images += 1

            pixels = np.array(pixels)
            pixels = pixels / 255 # Normalize data 
            pixels = np.where(pixels > 0.5, 1, 0) # Transform to binary
            return pixels.tolist(), label

    def __del__(self):
        # Close the files if the object is destroyed
        if hasattr(self, 'images_file') and not self.images_file.closed:
            self.images_file.close()
        if hasattr(self, 'labels_file') and not self.labels_file.closed:
            self.labels_file.close()

    def to_list(self):
        return list(self)
    
    def to_numpy(self):
        dataset = self.to_list()
        X = np.array([item[0] for item in dataset])
        Y = np.array([item[1] for item in dataset])
        return X, Y


def test():
    # Read the 10 first images from the MNIST dataset
    dataset = MNIST("./data/t10k-images-idx3-ubyte.gz", "./data/t10k-labels-idx1-ubyte.gz", 10)

    # Iterate the dataset
    images = []
    for img, label in dataset:
        images.append((img, label))

    # Show the result
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, len(images), figsize=(10, 2))
    for i in range(len(images)):
        img = np.array(images[i][0]).reshape(28, 28)
        axes[i].set_title(f"{images[i][1]}")
        axes[i].imshow(img, cmap="gray")  # Redimensiona cada imagen a 28x28
        axes[i].axis("off")  # Oculta los ejes

    plt.show()

if __name__ == "__main__":
    test()
    