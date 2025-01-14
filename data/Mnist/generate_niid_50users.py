# from sklearn.datasets import fetch_mldata
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
train_path = './data/Mnist/train/mnist_train.json'
test_path = './data/Mnist/test/mnist_test.json'
public_path = './data/Mnist/public/public_data.json'
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

labels = np.zeros((10,), dtype=int)

# print(mnist.data)
# for k in mnist.target:
#     for i in trange(10):
#         if k == i:
#             labels[i]+=1
# print(labels[0])

for i in trange(10):

    idx = (mnist.target) == i

    # mnist_data.append(mnist.data[idx])
    mnist_data.append(mnist.data[idx][:int(len(mnist.data[idx])*0.995)])
    public_mnist_data.append(
        mnist.data[idx][int(len(mnist.data[idx]) * 0.995):])


#
#
# print("all num", mnist_data)


print("\nNumb samples of each label:\n", [len(v) for v in mnist_data])
print("\nNumb samples of each label:\n", [len(k) for k in public_mnist_data])
# print(public_mnist_data)
###### CREATE USER DATA SPLIT #######
# Assign 100 samples to each user
X = [[] for _ in range(NUM_USERS)]
y = [[] for _ in range(NUM_USERS)]
X_public = []
y_public = []
idx = np.zeros(10, dtype=np.int64)
for user in range(NUM_USERS):
    for j in range(NUM_LABELS):  # 3 labels for each users
        #l = (2*user+j)%10
        l = (user + j) % 10
        # print("L:", l)
        X[user] += mnist_data[l][idx[l]:idx[l]+10].values.tolist()
        y[user] += (l*np.ones(10)).tolist()
        idx[l] += 10

print("IDX1:", idx)  # counting samples for each labels

# Assign remaining sample by power law
user = 0
# props = np.random.lognormal(
#     0, 2., (10, NUM_USERS, NUM_LABELS))  # last 5 is 5 labels
# props = np.array([[[len(v)-NUM_USERS]] for v in mnist_data]) * \
#     props/np.sum(props, (1, 2), keepdims=True)

props = np.random.lognormal(
    0, 1., (10, NUM_USERS, NUM_LABELS))  # last 5 is 5 labels
props = np.array([[[len(v)-100]] for v in mnist_data]) * \
    props/np.sum(props, (1, 2), keepdims=True)
# props = np.array([[[len(v)]] for v in mnist_data]) * \
#     props/np.sum(props, (1, 2), keepdims=True)


# print("here:",props/np.sum(props,(1,2), keepdims=True))
# props = np.array([[[len(v)-100]] for v in mnist_data]) * \
#    props/np.sum(props, (1, 2), keepdims=True)
#idx = 1000*np.ones(10, dtype=np.int64)
# print("here2:",props)
for user in trange(NUM_USERS):
    for j in range(NUM_LABELS):  # 4 labels for each users
        # l = (2*user+j)%10
        l = (user + j) % 10
        num_samples = int(props[l, user//int(NUM_USERS/10), j])
        # num_samples_public = int(num_samples) - int(num_samples*0.4)

        num_samples = int(num_samples * 0.4)  # Scale down number of samples

        # num_samples = int(num_samples)  # Scale down number of samples
        # num_samples = int(props[l, user//int(NUM_USERS/10), j])
        # numran1 = random.randint(10, 100)
        # numran2 = random.randint(1, 10)
        # num_samples = (num_samples) * numran2 + numran1 + 150
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
    # print(len(public_mnist_data[k]))
print("length", len(X_public))
print("length", len(y_public))

# print("check len os user:", user, j,
#       "len data", len(X[user]), num_samples)

print("IDX2:", idx)  # counting samples for each labels

# Create data structure
train_data = {'users': [], 'user_data': {}, 'num_samples': []}
test_data = {'users': [], 'user_data': {}, 'num_samples': []}
public_data = {'public_data': {}, 'num_samples_public': []}
all_samples = []
public_data["public_data"] = {'x': X_public, 'y': y_public}
public_data['num_samples_public'].append(len(y_public))
# Setup 5 users
# for i in trange(5, ncols=120):
# Setup 5 users
# for i in trange(5, ncols=120):
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

    test_data['users'].append(uname)
    test_data["user_data"][uname] = {'x': X_test, 'y': y_test}
    test_data['num_samples'].append(len(y_test))
    all_samples.append(train_len + test_len)

print("Num_samples:", train_data['num_samples'])
print("Total_samples:", sum(
    train_data['num_samples'] + test_data['num_samples']))
print("Numb_testing_samples:", test_data['num_samples'])
print("Total_testing_samples:", sum(test_data['num_samples']))
print("Median of data samples:", np.median(all_samples))

with open(train_path, 'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)
with open(public_path, 'w') as outfile:
    json.dump(public_data, outfile)
print("Finish Generating Samples")
