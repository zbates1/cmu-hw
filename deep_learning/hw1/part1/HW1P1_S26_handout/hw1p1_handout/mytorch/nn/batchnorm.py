import numpy as np


class BatchNorm1d:
    """
    Create your own mytorch.nn.BatchNorm1d!
    Read the writeup (Hint: Batch Normalization Section) for implementation details for the BatchNorm1d class.
    Hint: Read all the expressions given in the writeup and be CAREFUL to re-check your code.
    """

    def __init__(self, num_features, alpha=0.9):
        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during inference.
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        Forward pass for batch normalization.
        :param Z: batch of input data Z (N, num_features).
        :param eval: flag to indicate training or inference mode.
        :return: batch normalized data.

        Read the writeup (Hint: Batch Normalization Section) for implementation details for the BatchNorm1d forward.
        Note: The eval parameter indicate whether it's training phase or the inference phase of the problem.
        Check the values you need to recompute when eval = False.
        """
        self.Z = Z
        # self.N = None  # TODO: Calculate batch size
        self.N = Z.shape[0]
        # self.M = None  # TODO: Calculate mini-batch mean
        self.M = np.mean(Z, axis=0, keepdims=True)
        # self.V = None  # TODO: Calculate mini-batch variance
        self.V = np.var(Z, axis=0, keepdims=True)

        if eval == False:
            # training mode
            # self.NZ = None  # TODO: Calculate the normalized input Ẑ
            self.NZ = (Z - self.M) / np.sqrt(self.V + self.eps)
            # self.BZ = None  # TODO: Calculate the scaled and shifted for the normalized input Ẑ
            self.BZ = self.BW * self.NZ + self.Bb

            # self.running_M = None  # TODO: Calculate running mean
            self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M
            # self.running_V = None  # TODO: Calculate running variance
            self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V
        else:
            # inference mode
            # self.NZ = None  # TODO: Calculate the normalized input Ẑ using the running average for mean and variance
            self.NZ = (Z - self.running_M) / np.sqrt(self.running_V + self.eps)
            # self.BZ = None  # TODO: Calculate the scaled and shifted for the normalized input Ẑ
            self.BZ = self.BW * self.NZ + self.Bb

        return self.BZ

    def backward(self, dLdBZ):
        """
        Backward pass for batch normalization.
        :param dLdBZ: Gradient loss wrt the output of BatchNorm transformation for Z (N, num_features).
        :return: Gradient of loss (L) wrt batch of input batch data Z (N, num_features).

        Read the writeup (Hint: Batch Normalization Section) for implementation details for the BatchNorm1d backward.
        """
        # self.dLdBb = None  # TODO: Sum over the batch dimension.
        self.dLdBb = np.sum(dLdBZ, axis=0, keepdims=True)
        # self.dLdBW = None  # TODO: Scale gradient of loss wrt BatchNorm transformation by normalized input NZ.
        self.dLdBW = np.sum(dLdBZ * self.NZ, axis=0, keepdims=True)

        # dLdNZ = None  # TODO: Scale gradient of loss wrt BatchNorm transformation output by gamma (scaling parameter).
        dLdNZ = dLdBZ * self.BW

        # dLdV = None  # TODO: Compute gradient of loss backprop through variance calculation.
        dLdV = np.sum(dLdNZ * (self.Z - self.M) * (-0.5) * ((self.V + self.eps) ** (-1.5)), axis=0, keepdims=True)
        # dNZdM = None  # TODO: Compute derivative of normalized input with respect to mean.
        # Note: simplified since sum(Z - M) = 0
        dNZdM = -((self.V + self.eps) ** (-0.5))
        # dLdM = None  # TODO: Compute gradient of loss with respect to mean.
        dLdM = np.sum(dLdNZ * dNZdM, axis=0, keepdims=True) + dLdV * (-2 / self.N) * np.sum(self.Z - self.M, axis=0, keepdims=True)

        # dLdZ = None  # TODO: Compute gradient of loss with respect to the input.
        dLdZ = dLdNZ * ((self.V + self.eps) ** (-0.5)) + dLdV * (2 / self.N) * (self.Z - self.M) + dLdM * (1 / self.N)
        # raise NotImplemented  # TODO - What should be the return value?
        return dLdZ
