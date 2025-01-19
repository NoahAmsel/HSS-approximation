import numpy as np
from scipy.sparse.linalg import aslinearoperator as alo, LinearOperator


class HcatLinearOperator(LinearOperator):
    def __init__(self, A1: LinearOperator, A2: LinearOperator, dtype=None):
        assert A1.shape[0] == A2.shape[0]
        self.A1 = A1
        self.A2 = A2
        super().__init__(dtype, (A1.shape[0], A1.shape[1] + A2.shape[1]))
        self._init_dtype()

    def _matmat(self, other):
        return self.A1 @ other[:self.A1.shape[1]] + self.A2 @ other[self.A1.shape[1]:]

    def _adjoint(self):
        VcatLinearOperator(self.A1.T, self.A2.T)

    def toarray(self):
        return np.concatenate([self.A1.toarray(), self.A2.toarray()], axis=1)


class VcatLinearOperator(LinearOperator):
    def __init__(self, A1: LinearOperator, A2: LinearOperator, dtype=None):
        assert A1.shape[1] == A2.shape[1]
        self.A1 = A1
        self.A2 = A2
        super().__init__(dtype, (A1.shape[0] + A2.shape[0], A1.shape[1]))
        self._init_dtype()

    def _matmat(self, other):
        return np.concatenate((self.A1 @ other, self.A2 @ other))

    def _adjoint(self):
        HcatLinearOperator(self.A1.T, self.A2.T)

    def toarray(self):
        return np.concatenate([self.A1.toarray(), self.A2.toarray()], axis=0)


class Block22Matrix(VcatLinearOperator):
    def __init__(self, A11: LinearOperator, A12: LinearOperator, A21: LinearOperator, A22: LinearOperator):
        super().__init__(HcatLinearOperator(A11, A12), HcatLinearOperator(A21, A22))


class FourPartLens(LinearOperator):
    def __init__(self, U, A, V, D, dtype=None):
        """
        A factorization of the form UAV^T + D.
        Assumes nothing about the four pieces other than:
            1. They implement the LinearOperator interface
            2. They have a property called .size that says how many parameters they contain
            3. Their dimensions are compatible
        """
        assert U.shape[0] == D.shape[0]
        assert U.shape[1] == A.shape[0]
        assert A.shape[1] == V.T.shape[0]
        assert V.T.shape[1] == D.shape[1]
        self.U = U
        self.V = V
        self.A = A
        self.D = D
        super().__init__(dtype, D.shape)
        self._init_dtype()

    def _matmat(self, other):
        return self.U @ (self.A @ (self.V.T @ other)) + self.D @ other

    def _adjoint(self):
        type(self)(self.V, self.A.T, self.U, self.D.T)

    def toarray(self):
        return np.array(self.U @ (self.A @ self.V.T) + self.D)

    @property
    def size(self):
        return self.U.size + self.A.size + self.V.size + self.D.size
