import os
import sys
import gzip
import shutil
from six.moves import urllib

# mnist dataset
HOMEPAGE = "http://yann.lecun.com/exdb/mnist/"
MNIST_TRAIN_IMGS_URL = HOMEPAGE + "train-images-idx3-ubyte.gz"
MNIST_TRAIN_LABELS_URL = HOMEPAGE + "train-labels-idx1-ubyte.gz"
MNIST_TEST_IMGS_URL = HOMEPAGE + "t10k-images-idx3-ubyte.gz"
MNIST_TEST_LABELS_URL = HOMEPAGE + "t10k-labels-idx1-ubyte.gz"

# fashion-mnist dataset
HOMEPAGE = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
FASHION_MNIST_TRAIN_IMGS_URL = HOMEPAGE + "train-images-idx3-ubyte.gz"
FASHION_MNIST_TRAIN_LABELS_URL = HOMEPAGE + "train-labels-idx1-ubyte.gz"
FASHION_MNIST_TEST_IMGS_URL = HOMEPAGE + "t10k-images-idx3-ubyte.gz"
FASHION_MNIST_TEST_LABELS_URL = HOMEPAGE + "t10k-labels-idx1-ubyte.gz"

def download_and_uncompress_zip(URL, dataset_dir, force=False):
    filename = URL.split('/')[-1]
    fn = os.path.join(dataset_dir, filename)
    out_dir = os.path.splitext(fn)[0]

    def doanload_progress(count, block, total_size):
        sys.stdout.write("\r>> Download %s %.1f%%" %(fn, float(count * block) / float(total_size)*100.))
        sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(URL, fn, doanload_progress)

    with gzip.open(fn, 'rb') as f_in, open(out_dir, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
        

def start_download(dataset, save_to):

    if (not os.path.exists(save_to)):
        os.makedirs(save_to)

    if (dataset == 'mnist'):
        download_and_uncompress_zip(MNIST_TRAIN_IMGS_URL, save_to)
        download_and_uncompress_zip(MNIST_TRAIN_LABELS_URL, save_to)
        download_and_uncompress_zip(MNIST_TEST_IMGS_URL, save_to)
        download_and_uncompress_zip(MNIST_TEST_LABELS_URL, save_to)

    else:
        download_and_uncompress_zip(FASHION_MNIST_TRAIN_IMGS_URL, save_to)
        download_and_uncompress_zip(FASHION_MNIST_TRAIN_LABELS_URL, save_to)
        download_and_uncompress_zip(FASHION_MNIST_TEST_IMGS_URL, save_to)
        download_and_uncompress_zip(FASHION_MNIST_TEST_LABELS_URL, save_to)

if __name__ == '__main__':
    if sys.argv[1] == 'mnist':
        start_download('mnist', './mnist/')
    else:
        start_download('fashion_mnist', './fashion_mnist/')
