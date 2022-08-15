

from datetime import date
import numpy as np
# from sklearn.datasets import fetch_openml
# from tqdm import trange


class Father():
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.c = 1

    def dev(self):
        print(self.c)
        return self.a - self.b


class Son(Father):
    def __init__(self, a, b, c=11):
        Father.__init__(self, a, b)
        self.c = c

    def add(self):
        return self.a+self.b

    def compare(self):
        if self.c > (self.a+self.b):
            return True
        else:
            return False

# son = Son(1, 2)
# print(son.dev())
# a = [1, 2, 3, 4]
# print(list(enumerate(a))[0])
# id = 1
# test_acc = 80
# y_size = 200
# loss = 0.003
# print("User ", id, ", ALl test dataset Accuracy:",
#       test_acc / y_size, ", All test dataset Loss:", loss)


# private_data = ((1, 2), (3, 4), (5, 6))
# public_data = [(1, (2, 6)), (2, (5, 7)), (3, (3, 8)), (4, (1, 0))]
# public_data = public_data[1:]

# for [local_batch_idx, (X, y)], [batch_idx, (X_public, y_public)] in zip(enumerate(private_data), public_data):
#     print("local batch id: ", local_batch_idx,
#           " public batch ", batch_idx)
# print(aa[6:])
# print(aa)

# dic = dict()
# dic["1"] = 222
# dic["2"] = 546
# for i in dic:
#     print(dic[i], i)


# a = np.zeros((10,))
# print(a)
# print(a.shape)
# b = np.zeros(10)
# print(b)
# print(b.shape)

# mnist = fetch_openml('mnist_784', version=1, cache=True)

# # fetch_openml() returns targets as strings
# mnist.target = mnist.target.astype(np.int8)

# mu = np.mean(mnist.data.astype(np.float32), 0)
# sigma = np.std(mnist.data.astype(np.float32), 0)
# mnist.data = (mnist.data.astype(np.float32) - mu)/(sigma+0.001)

# mnist_data = []
# public_mnist_data = []
# # 10x1
# labels = np.zeros((10,), dtype=int)

# # split a small part from the Mnist dataset as a public dataset
# for i in trange(10):
#     idx = (mnist.target) == i
#     # mnist_data.append(mnist.data[idx])
#     mnist_data.append(mnist.data[idx][:int(len(mnist.data[idx])*0.995)])
#     public_mnist_data.append(
#         mnist.data[idx][int(len(mnist.data[idx]) * 0.995):])


# print("\nNumb samples of each label:\n", [len(v) for v in mnist_data])
# print("\nNumb samples of each label:\n", [len(k) for k in public_mnist_data])

# idx = np.zeros(10, dtype=np.int64)
# print(idx)

props = np.random.lognormal(
    0, 1., (10, 50, 2))
# print(props[0])

mnist_data = [6868, 7837, 6955, 7105, 6789, 6281, 6841, 7256, 6790, 6923]

# r = np.array([[[v-100]] for v in mnist_data])
# print(r[0])
# print(r[0]*props[0])

# print(np.sum(props, (1, 2), keepdims=True))

# te = np.array([[[1, 2], [1, 2]], [[3, 4], [4, 5]]])
# add dimension 2 and 3
# print(np.sum(te, (1, 2), keepdims=True))

# props = np.array([[[v-100]] for v in mnist_data]) * \
#     props/np.sum(props, (1, 2), keepdims=True)

# print(props[0])

d = np.array([0 for i in range(10)])

mmm = np.array([[1, 2, 0, 3, 0, 5, 0, 6, 8, 0],
                [1, 0, 2, 0, 0, 5, 0, 6, 0, 4],
                [0, 2, 0, 3, 3, 0, 2, 0, 8, 2]])

a = np.array([1, 2, 0, 3, 0, 5, 0, 6, 8, 0])
b = np.array([1, 0, 2, 0, 0, 5, 0, 6, 0, 4])
c = np.array([0, 2, 0, 3, 3, 0, 2, 0, 8, 2])
d += np.array([1 if i == 0 else 0 for i in a])
d += np.array([1 if i == 0 else 0 for i in b])
d += np.array([1 if i == 0 else 0 for i in c])
print(d)
print(np.divide(mmm, d))
