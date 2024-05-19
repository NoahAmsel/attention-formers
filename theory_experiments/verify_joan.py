from functools import cache

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import gegenbauer


class Gegenbauer:
    def __init__(self, dim):
        assert dim > 2, "Scipy not set up for alpha = 0. Need to rewrite as a special case to use Legendre polynomials."
        self.dim = dim
        self.limit = 600

    def quad_ud(self, f):
        alpha = (self.dim-3)/2
        main = quad(f, -1, 1, limit=self.limit, weight='alg', wvar=(alpha, alpha))[0]
        normalization = quad(lambda x: 1., -1, 1, limit=self.limit, weight='alg', wvar=(alpha, alpha))[0]
        return main / normalization

    def inner_ud(self, f, g):
        def fg(x): return f(x)*g(x)
        return self.quad_ud(fg)

    def norm_ud(self, f):
        return np.sqrt(self.inner_ud(f, f))

    def geg(self, deg):
        alpha = self.dim/2-1
        return gegenbauer(deg, alpha)

    @cache
    def norm_geg(self, deg):
        return self.norm_ud(self.geg(deg))

    def normalized_geg(self, deg):
        unnormalized = self.geg(deg)
        norm = self.norm_geg(deg)
        return lambda x: unnormalized(x) / norm


class GegenbauerTransform(Gegenbauer):
    def __init__(self, dim, fun, parity=None):
        super().__init__(dim)
        self.fun = fun
        self.parity = parity

    @cache
    def coeff(self, deg):
        if ((self.parity == "odd") and (deg % 2 == 0)) or ((self.parity == "even") and (deg % 2 == 1)):
            return 0
        else:
            return self.inner_ud(self.fun, self.normalized_geg(deg))


class GegenbauerInverseTransform(Gegenbauer):
    def __init__(self, dim, coeffs, use_normalized_geg=True):
        super().__init__(dim)
        self.coeffs = coeffs
        self.use_normalized_geg = use_normalized_geg

    def __call__(self, x):
        if self.use_normalized_geg:
            gegfun = self.normalized_geg
        else:
            gegfun = self.geg
        return sum(coeff * gegfun(deg)(x) for deg, coeff in enumerate(self.coeffs))


class HeadVsTarget(GegenbauerInverseTransform):
    def __init__(self, dim, nterms):
        sign = GegenbauerTransform(dim, np.sign, 'odd')
        def features(x): return (np.sqrt(2)/np.pi) * np.arcsin(x)
        feature_expansion = GegenbauerTransform(dim, features, 'odd')
        super().__init__(
            dim,
            [
                # correct for the fact that Joan assumes Gegenbauer polynomials are normalized
                # to have the value 1 at 1, but scipy's gegenbauer polynomials are not 
                sign.coeff(deg) * feature_expansion.coeff(deg) / sign.geg(deg)(1)
                for deg in range(nterms)
            ],
            use_normalized_geg=False
        )

    def __call__(self, theta):
        x = np.cos(theta)
        return (1/2) + (1/np.sqrt(2)) * super().__call__(x)


if __name__ == "__main__":
    # dim = 20
    for dim in [3, 6, 10, 20, 40, 80]:
        G = GegenbauerTransform(dim, np.sign, 'odd')
        degs = np.arange(30)
        etas = np.array([G.coeff(deg) for deg in degs])
        # etas = np.array([inner_ud(dim, np.sign, geg(dim, deg)) / norm_ud(dim, geg(dim, deg)) for deg in degs])
        # plt.plot(degs, etas)
        plt.plot(degs[1::2], np.abs(1/etas[1::2]))
