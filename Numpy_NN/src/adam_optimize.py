import numpy as np

class Adam:
    """Реализует Adam

    ---------
    Параметры
    ---------
    params
        Параметры, передаваемые из модели

    lr : float (default=3e-4)
        Learning rate

    beta_1 : float (default=0.9)
        Параметр beta_1

    beta_2 : float (default=0.999)
        Параметр beta_2

    eps : float (default=1e-8)
        Параметр eps

    alpha1 : float (default=None)
        Если не None, то применяет l_1 регуляризацию
        с параметром alpha_1

    alpha2 : float (default=None)
        Если не None, то применяет l_2 регуляризацию
        с параметром alpha_2
    """

    def __init__(self, params, lr=3e-4, beta_1=0.9, beta_2=0.999, eps=1e-8,
                 alpha1=None, alpha2=None):
        # super().__init__(params, lr)
        self.params = list(params)
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.t = 0

        for param in self.params:
            param.m = np.zeros(param.shape)
            param.v = np.zeros(param.shape)

    def zero_grad(self):
        for param in self.params:
            param.grads = np.zeros(param.shape)

    def step(self):
        # TODO: Реализовать шаг Adam
        # Аналогичная реализация классического градиентоного спуска реализована в gd_optimizer.py
        self.t += 1
        for param in self.params:
            grads = np.zeros(param.grads.shape)
            if not (self.alpha1 is None):
                grads += self.alpha1 * np.sign(param.params)
            if not (self.alpha2 is None):
                grads += self.alpha2 * param.params
            grads += param.grads
            param.m = self.beta_1 * param.m + (1 - self.beta_1) * grads
            param.v = self.beta_2 + param.v + (1 - self.beta_2) * (grads ** 2)
            
            m_hat = param.m / (1 - self.beta_1 ** self.t)
            v_hat = param.v / (1 - self.beta_2 ** self.t)
            param.params = param.params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
