import math
import random
from typing import Callable

import cv2
import numpy as np
from tqdm import trange

DISTINCT_COLORS = [(252, 186, 3), (45, 252, 3), (3, 252, 227), (0, 26, 255), (242, 0, 255), (255, 0, 64)]


class proposed_alpha1:
    def __init__(self, c1=80, c2=1):
        self.c1 = c1
        self.c2 = c2
        self.t = 0

    def __call__(self, t=None):
        if t is not None:
            self.t = t - 1
        self.t += 1
        return self.c1 * 0.9 ** self.t + self.c2

    def reset(self):
        self.t = 0


class proposed_alpha2:
    def __init__(self, k, c1=80):
        self.c1 = c1
        self.k = k
        self.t = 0

    def __call__(self, t=None):
        if t is not None:
            self.t = t - 1
        self.t += 1
        return self.c1 * 0.9 ** self.t + 1 / self.k

    def reset(self):
        self.t = 0


class constant_alpha:
    def __init__(self, const):
        self.const = const

    def __call__(self, t=None):
        return self.const

    def reset(self):
        pass


class simple_MRF_segmentation:
    def __init__(self, n_segments, alpha: Callable[[int], float], beta=1, random_state=None):
        self.img = np.array([])
        self.alpha = alpha
        self.beta = beta
        self.__mu = np.array([0] * n_segments)
        self.__sigma = np.array([0] * n_segments)
        self.__labels = np.array([])
        self.__correct_mu_sigma = False
        self.__correct_labels = False
        self.n_segments = n_segments
        if random_state is not None:
            random.seed(random_state)

    @property
    def mu(self):
        if self.__correct_mu_sigma:
            return self.__mu
        self.__mu, self.__sigma = self.__calculate_mu_sigma()
        self.__correct_mu_sigma = True
        return self.__mu

    @property
    def sigma(self):
        if self.__correct_mu_sigma:
            return self.__sigma
        self.__mu, self.__sigma = self.__calculate_mu_sigma()
        self.__correct_mu_sigma = True
        return self.__sigma

    @property
    def labels(self):
        return self.__labels

    @labels.setter
    def labels(self, value):
        self.__labels = np.array(value, dtype=np.uint8)
        self.__correct_mu_sigma = False

    def __calculate_mu_sigma(self):
        mu = []
        sigma = []
        for c in range(self.n_segments):
            class_features = self.img[self.__labels == c]
            mu.append(np.mean(class_features))
            sigma.append(np.std(class_features))
        return np.array(mu), np.array(sigma)

    @staticmethod
    def Er(labels, beta=1):
        kernels = [np.array([[0, -1, 0],
                             [0, 1, 0],
                             [0, 0, 0]], dtype=np.float32),

                   np.array([[0, 0, -1],
                             [0, 1, 0],
                             [0, 0, 0]], dtype=np.float32),

                   np.array([[0, 0, 0],
                             [0, 1, -1],
                             [0, 0, 0]], dtype=np.float32),

                   np.array([[0, 0, 0],
                             [0, 1, 0],
                             [0, 0, -1]], dtype=np.float32)]
        Er = 0
        for kernel in kernels:
            try:
                labels = np.array(labels, dtype=np.uint8)
                tmp = cv2.filter2D(labels, -1, kernel)
            except Exception as e:
                print(labels)
                print(kernel)
                print(labels.dtype)
                print(kernel.dtype)
                raise e
            '''
            tmp = 0 iff two corresponding pixels are equal
            tmp!= 0 iff two corresponding pixels are not equal
            as described in paper delta must be -1 if corresponding pixels are equal otherwise 1
            '''
            tmp[tmp != 0] = 2
            tmp -= 1
            tmp *= beta
            # if Er is not None:
            #     Er += tmp
            # else:
            #     Er = tmp
            Er += np.sum(tmp)
        return np.sum(Er)

    def Ef(self):
        Ef = 0
        for mu, sigma in zip(self.mu, self.sigma):
            Ef += np.sum(self.img - mu) ** 2 / (2 * sigma ** 2) + math.log((2 * math.pi) ** 0.5 * sigma + 0.01)
        return Ef

    def E(self, t, labels):
        return self.Ef() + self.alpha(t) * self.Er(labels)

    def label_image(self, mu, sigma):
        Ps = []
        assert len(mu) == len(mu) == self.n_segments
        '''
        Ps is the probability of each class for that specified pixel
        for example Ps[0] is a matrix showing how much it is probable to see label Zero for each pixel
        '''
        for mu_, sigma_ in zip(mu, sigma):
            Ps.append(1 / math.sqrt(2 * math.pi * sigma_ ** 2) * np.exp(-(self.img - mu_) ** 2 / (2 * sigma_ ** 2)))
        return np.argmax(np.stack(Ps, axis=2), axis=2)

    def segment(self, img, n_iter=150, save_name: str = None):
        self.img = img
        self.__correct_mu_sigma = False
        self.__labels = np.random.randint(0, self.n_segments, size=img.shape)
        self.__mu = np.random.random(size=(self.n_segments,))
        self.__sigma = np.random.random(size=(self.n_segments,))
        self.labels = self.label_image(self.__mu, self.__sigma)

        for t in trange(1, n_iter + 1):
            self.metropolise(n_iter=100, t=t)

        if save_name is not None:
            if not save_name.endswith(('png', 'jpg')):
                save_name += 'jpg'
            self.save_image(save_name)
        return self.labels

    def metropolise(self, n_iter, t, C=2):
        for i in range(1, n_iter + 1):
            # print(i)
            # previous_mu_sigma = (self.mu.copy(), self.sigma.copy())
            previous_energy = self.E(t, self.labels)
            mu_star = self.mu + random.random()
            sigma_star = self.sigma + random.random()
            labels_star = self.label_image(mu_star, sigma_star)
            alpha = math.exp((previous_energy * math.log(i) - self.E(t, labels_star) * math.log(i + 1)) / C)
            if alpha > random.random():
                self.__sigma = sigma_star
                self.__mu = mu_star
                self.__labels = labels_star

    def save_image(self, name):
        cv2.imwrite(name, np.stack(np.vectorize(lambda x: DISTINCT_COLORS[x])(self.labels)).astype(np.uint8).T)


if __name__ == '__main__':
    fig5 = cv2.imread('fig5.jpg', 0)
    fig9 = cv2.imread('fig9.jpg', 0)
    SMS5 = simple_MRF_segmentation(n_segments=2, alpha=proposed_alpha1(), random_state=0)
    SMS9 = simple_MRF_segmentation(n_segments=3, alpha=proposed_alpha1(), random_state=0)
    # SMS9.segment(fig9, save_name='fig9_out.jpg')
    SMS5.segment(fig5,save_name='fig5_out.jpg')