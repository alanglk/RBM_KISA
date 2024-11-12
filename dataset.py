import gzip

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
    
    """
    
    def __init__(self, path: str, N: int = None) -> None:
        """
            INPUT:
                - path: path of the dataset images
                - N: if not provided, all the images in the input file are read.
                    Else, N images are read.
        """

        self.path = path

        # Open with gzip to uncompress the file
        self.file = gzip.open(self.path, "rb")
        self.file.read(4) # Ignore the magic number

        # Obtain metadata
        self.number_of_items = int.from_bytes(self.file.read(4), byteorder="big") # Total number of images
        self.row_num = int.from_bytes(self.file.read(4), byteorder="big")    # row pixels of image
        self.col_num = int.from_bytes(self.file.read(4), byteorder="big")    # col pixels of image
        
        self.N = self.number_of_items if N is None else N # Number of items to be read
        self.num_bytes_per_image = self.col_num * self.row_num
        self.returned_images = 0    # Number of read images

        print(f"[MNIST]     image size: {self.col_num}x{self.row_num} total dataset images: {self.number_of_items} images to be read: {self.N}")

        # Set the cursor to the init of the first image
        self.file.seek(16)

    def __iter__(self):
        """Create an iterator for the MNIST dataset"""
        return self

    def __next__(self):
        """Return the next image in the dataset"""
        if self.returned_images >= self.N:
            self.file.close()
            raise StopIteration
        else:
            pixels = []
            for _ in range(self.num_bytes_per_image):
                data = self.file.read(1)

                if not data:
                    print(f"[MNIST]     ERROR -> This should not be reached")
                    self.file.close()
                    raise StopIteration
                
                pixels.append(int.from_bytes(data, byteorder="big"))
            self.returned_images += 1
            return pixels

    def __del__(self):
        # Close the file if the object is destroyed
        if hasattr(self, 'file') and not self.file.closed:
            self.file.close()

def test():
    # Read the 10 first images from the MNIST dataset
    dataset = MNIST("./data/t10k-images-idx3-ubyte.gz", 10)

    # Iterate the dataset
    images = []
    for img in dataset:
        images.append(img)

    # Show the result
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, len(images), figsize=(10, 2))
    for i in range(len(images)):
        img = np.array(images[i]).reshape(28, 28)
        axes[i].imshow(img, cmap="gray")  # Redimensiona cada imagen a 28x28
        axes[i].axis("off")  # Oculta los ejes

    plt.show()

if __name__ == "__main__":
    test()
    