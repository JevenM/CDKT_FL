from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tqdm import trange
import numpy as np
import random
import json
import os

random.seed(1)
np.random.seed(1)

NUM_USERS = 50
NUM_LABELS = 2

# Setup directory for train/test data
train_path = './data/Mnist/data/train/mnist_train.json'
test_path = './data/Mnist/data/test/mnist_test.json'
public_path = './data/Mnist/data/public/public_data.json'
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(public_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Get MNIST data, normalize, and divide by level
# mnist = fetch_mldata('MNIST original', data_home='./data')
# mnist = fetch_openml('MNIST original', data_home='./data')
mnist = fetch_openml('mnist_784', version=1, cache=True)

# fetch_openml() returns targets as strings
mnist.target = mnist.target.astype(np.int8)

mu = np.mean(mnist.data.astype(np.float32), 0)
sigma = np.std(mnist.data.astype(np.float32), 0)
mnist.data = (mnist.data.astype(np.float32) - mu)/(sigma+0.001)

mnist_data = []
public_mnist_data = []
# 10��1
# labels = np.zeros((10,), dtype=int)

# split a small part from the Mnist dataset as a public dataset
for i in trange(10):
    idx = (mnist.target) == i
    # mnist_data.append(mnist.data[idx])
    mnist_data.append(mnist.data[idx][:int(len(mnist.data[idx])*0.995)])
    public_mnist_data.append(
        mnist.data[idx][int(len(mnist.data[idx]) * 0.995):])

# public: [6868, 7837, 6955, 7105, 6789, 6281, 6841, 7256, 6790, 6923], total: 69645
print("\nNumb samples of each label:\n", [len(v) for v in mnist_data])
# public: [35, 40, 35, 36, 35, 32, 35, 37, 35, 35], total: 355
print("\nNumb samples of each label in public dataset:\n",
      [len(k) for k in public_mnist_data])

# ------------ CREATE USER DATA SPLIT ------------------

X = [[] for _ in range(NUM_USERS)]
y = [[] for _ in range(NUM_USERS)]
X_public = []
y_public = []
# 10x1
idx = np.zeros(10, dtype=np.int64)

# Assign 100 samples to each user
for user in range(NUM_USERS):
    for j in range(NUM_LABELS):  # 2 labels for each users
        lei = (user + j) % 10
        # print("L:", lei)
        X[user] += mnist_data[lei][idx[lei]:idx[lei]+10].values.tolist()
        y[user] += (lei*np.ones(10)).tolist()
        idx[lei] += 10

# counting samples for each labels: [100 100 100 100 100 100 100 100 100 100]
print("Each class has sample number(IDX1):", idx)

# Assign remaining sample by power law
user = 0

# matrix 10*50*2
props = np.random.lognormal(
    0, 1., (10, NUM_USERS, NUM_LABELS))
# 10*50*2
props = np.array([[[len(v)-100]] for v in mnist_data]) * \
    props/np.sum(props, (1, 2), keepdims=True)

# add num_samples samples for each user
for user in trange(NUM_USERS):
    for j in range(NUM_LABELS):  # 2 labels for each users
        # leibie index
        l = (user + j) % 10
        num_samples = int(props[l, user//int(NUM_USERS/10), j])
        # num_samples_public = int(num_samples) - int(num_samples*0.4)

        num_samples = int(num_samples * 0.4)  # Scale down number of samples

        if(NUM_USERS <= 20):
            num_samples = num_samples * 2

        if idx[l] + num_samples < len(mnist_data[l]):
            X[user] += mnist_data[l][idx[l]:idx[l]+num_samples].values.tolist()
            y[user] += (l*np.ones(num_samples)).tolist()
            idx[l] += num_samples
            # X_public += mnist_data[l][idx[l]:idx[l]+num_samples_public:].values.tolist()
            # y_public += (l * np.ones(num_samples_public)).tolist()


for k in trange(10):
    X_public += public_mnist_data[k].values.tolist()
    y_public += (k*np.ones(len(public_mnist_data[k]))).tolist()

# 335, 335
print("length of X_public:", len(X_public),
      "length of y_public:", len(y_public))


# counting samples for each labels: [326 291 370 476 373 273 292 414 451 508], all 3774
print("Each class has sample number(IDX2):", idx)

# Create data structure
train_data = {'users': [], 'user_data': {}, 'num_samples': []}
test_data = {'users': [], 'user_data': {}, 'num_samples': []}
public_data = {'public_data': {}, 'num_samples_public': []}
public_data["public_data"] = {'x': X_public, 'y': y_public}
public_data['num_samples_public'].append(len(y_public))

all_samples = []

for i in range(NUM_USERS):
    uname = i
    X_train, X_test, y_train, y_test = train_test_split(
        X[i], y[i], train_size=0.8, stratify=y[i])
    # X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.75, stratify=y[i])
    num_samples = len(X[i])
    train_len = int(0.8*num_samples)  # Test 80%
    test_len = num_samples - train_len

    train_data["user_data"][uname] = {'x': X_train, 'y': y_train}
    train_data['users'].append(uname)
    train_data['num_samples'].append(len(y_train))
    # print("user:", uname, " train_len:", train_len)

    test_data['users'].append(uname)
    test_data["user_data"][uname] = {'x': X_test, 'y': y_test}
    test_data['num_samples'].append(len(y_test))
    all_samples.append(train_len + test_len)

# [132, 53, 26, 129, 28, 27, 81, 39, 48, 46, 64, 21, 84, 38, 34, 33, 89, 98, 29, 122, 48, 118, 26, 76, 50, 105, 36, 27, 64, 64, 31, 54, 61, 78, 90, 39, 81, 51, 180, 50, 29, 36, 52, 77, 36, 33, 60, 47, 28, 49]
print("Train data num_samples:", train_data['num_samples'])
# [34, 14, 7, 33, 8, 7, 21, 10, 13, 12, 16, 6, 21, 10, 9, 9, 23, 25, 8, 31, 13, 30, 7, 19, 13, 27, 10, 7, 17, 16, 8, 14, 16, 20, 23, 10, 21, 13, 46, 13, 8, 10, 13, 20, 9, 9, 15, 12, 8, 13]
print("Test data num_samples:", test_data['num_samples'])

# 3774
print("50 user Total_samples:", sum(
    train_data['num_samples'] + test_data['num_samples']))
# 777
print("Total_testing_samples:", sum(test_data['num_samples']))
# 2997
print("Total_training_samples:", sum(train_data['num_samples']))

# 63.0 the same as paper
print("Median of data samples(train+test):", np.median(all_samples))

with open(train_path, 'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)
with open(public_path, 'w') as outfile:
    json.dump(public_data, outfile)
print("Finish Generating Samples")
