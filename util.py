import os
from skimage import data

ROOT_PATH = "/Users/isohrab/Documents/tensorflow/trafficSigns"

def load_data(data_path):
    directories = [d for d in os.listdir(data_path)
                   if os.path.isdir(os.path.join(data_path, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_path,d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(data.imread(f))
            labels.append(int(d))
    return images, labels

train_data_dir = os.path.join(ROOT_PATH, "data/Training")
test_data_dir = os.path.join(ROOT_PATH, "data/Testing")

images_test, labels_test = load_data(test_data_dir)
images_train, labels_train = load_data(train_data_dir)



from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)
data.train.next_batch()
