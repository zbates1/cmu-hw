import numpy as np


class MSELoss:
    def forward(self, A, Y):
        """
        Calculate the Mean Squared error (MSE)
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss (scalar)

        Read the writeup (Hint: MSE Loss Section) for implementation details for below code snippet.
        """
        self.A = A
        self.Y = Y
        # self.N = None  # TODO
        self.N = A.shape[0]
        # self.C = None  # TODO
        self.C = A.shape[1]
        # se = None  # TODO
        se = (A - Y) * (A - Y)
        # sse = None  # TODO
        ones_N = np.ones((self.N, 1))
        ones_C = np.ones((self.C, 1))
        sse = ones_N.T @ se @ ones_C
        # mse = None  # TODO
        mse = sse / (self.N * self.C)
        # raise NotImplemented  # TODO - What should be the return value?
        return mse

    def backward(self):
        """
        Calculate the gradient of MSE Loss wrt model output A.
        :Return: Gradient of loss L wrt model output A.

        Read the writeup (Hint: MSE Loss Section) for implementation details for below code snippet.
        """
        # dLdA = None
        dLdA = 2 * (self.A - self.Y) / (self.N * self.C)
        # raise NotImplemented  # TODO - What should be the return value?
        return dLdA


class CrossEntropyLoss:
    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss (XENT)
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss (scalar)

        Read the writeup (Hint: Cross-Entropy Loss Section) for implementation details for below code snippet.
        Hint: Read the writeup to determine the shapes of all the variables.
        Note: Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        # self.N = None  # TODO
        self.N = A.shape[0]
        # self.C = None  # TODO
        self.C = A.shape[1]

        # Ones_C = None  # TODO
        Ones_C = np.ones((self.C, 1), dtype="f")
        # Ones_N = None  # TODO
        Ones_N = np.ones((self.N, 1), dtype="f")

        # self.softmax = None  # TODO
        # Numerical stability: subtract max from A
        A_stable = A - np.max(A, axis=1, keepdims=True)
        exp_A = np.exp(A_stable)
        self.softmax = exp_A / np.sum(exp_A, axis=1, keepdims=True)

        # crossentropy = None  # TODO
        crossentropy = (-Y * np.log(self.softmax)) @ Ones_C  # (N, 1)
        # sum_crossentropy_loss = None  # TODO
        sum_crossentropy_loss = Ones_N.T @ crossentropy  # scalar
        mean_crossentropy_loss = sum_crossentropy_loss / self.N

        # raise NotImplemented  # TODO - What should be the return value?
        return mean_crossentropy_loss

    def backward(self):
        """
        Calculate the gradient of Cross-Entropy Loss wrt model output A.
        :Return: Gradient of loss L wrt model output A.

        Read the writeup (Hint: Cross-Entropy Loss Section) for implementation details for below code snippet.
        """
        # dLdA = None  # TODO
        dLdA = (self.softmax - self.Y) / self.N
        # raise NotImplemented  # TODO - What should be the return value?
        return dLdA
