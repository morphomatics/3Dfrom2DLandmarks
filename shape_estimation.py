"""
This file is part of the publication:
M. Paskin, D. Baum, Mason N. Dean, Christoph von Tycowicz (2022)
A Kendall Shape Space Approach to 3D Shape Estimation from 2D Landmarks -- Source Code and Data
http://dx.doi.org/10.12752/8730
"""

from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Stiefel, Euclidean
from pymanopt.optimizers import SteepestDescent

from morphomatics.manifold import Kendall
from morphomatics.manifold import ManoptWrap # creates JAX backend for pymanopt

def frechetMean(mfd, pointset, weights):
    """
    Estimate weighted Frechet mean via recursive method.

    Parameters
    ----------
    mfd : Manifold
        Shape space.
    pointset : array-like, shape=[N, ...]
        Collection of sample points on mfd.
    weights : array-like, shape=[N]
        Weights.

    Returns
    -------
    mean : array-like, shape=[N, ...]
        Weighted Frechet mean.
    """
    # number of points
    num = np.size(weights)

    weights = weights / weights.sum()
    idx = jnp.argsort(-weights)
    weights = weights[idx]
    t = weights / jnp.cumsum(weights)

    loop = lambda i, mean: mfd.connec.geopoint(mean, pointset[idx[i]], t[i])
    return jax.lax.fori_loop(1, num, loop, pointset[0])


class ShapeEstimation:
    """
    This class implements a routine for 3D shape estimation from 2D landmarks
    presented in [1].

    The estimation problems is regularized by employing prior knowledge learned from
    examples of the target's object class. In particular, the 3D shape of the target
    is estimated by interpolating through a set of pre-defined 3D shapes, called
    training shapes.

    Parameters
    ----------
    W : array-like, shape=[k, 2]
        Projected 2D landmark positions of target shape.
    B : array-like, shape=[N, k, 3]
        Set of N training shapes in 3D.

    References
    ----------
    .. [1] M. Paskin, D. Baum, Mason N. Dean, Christoph von Tycowicz (2022).
    A Kendall Shape Space Approach to 3D Shape Estimation from 2D Landmarks.
    In: European Conference on Computer Vision. Springer
    """

    def __init__(self, W, B):
        self.W = W

        self.inputShapes = jax.vmap(Kendall.project)(B)
        self.numShapes = B.shape[0]
        self.numLandmarks = W.shape[1]
        
        #weights for the linear combination of basis shapes (vector "c")
        self.weights = None
        self.rotation = None
        self.frechetMean = None

    @partial(jax.jit, static_argnums=0)
    def mean(self, B, c):
        mfd = Kendall(shape=B[0].shape)
        fmean = frechetMean(mfd, B, c)
        return mfd.wellpos(B[0], fmean)

    @partial(jax.jit, static_argnums=0)
    def error(self, W, R, mean):
        nu = lambda s: s / jnp.linalg.norm(s)
        return 0.5 * jnp.sum((W - nu(mean @ R)) ** 2)

    def optimize(self, maxIter=100, tol=1e-6, verbose=0):
        """
        Alternating optimization for 3D-from-2D problem.

        Parameters
        ----------
        maxIter : int
            Number of maximal (outer) iterations.
            Optional, default: 100.
        tol : array-like, shape=[N, ...]
            Tolerance for convergence check.
            Optional, default: 1e-6.
        verbose : int
            Verbosity level: 0 is silent, 2 is most verbose.
            Optional, default: 0

        Returns
        -------
        mean : array-like, shape=[N, ...]
            Weighted Frechet mean.
        """

        V = Stiefel(3, 2)

        # initial guess
        self.weights = jnp.ones(self.numShapes)
        # self.rotation = jnp.asarray(V.random_point())
        mean = self.mean(self.inputShapes, self.weights)
        U, S, Vt = jax.numpy.linalg.svd(mean.T @ self.W)
        self.rotation = U[:, :2] @ Vt

        # define reprojection error in terms of R and c
        error = lambda R, c: self.error(self.W, R, self.mean(self.inputShapes, c))

        it = 0
        newCost = changeInSS_weights = 1e10
        cost = error(self.rotation, self.weights)

        # alternating optization
        for it in range(0, maxIter):

            epsilon = abs(newCost - cost)
            cost = newCost

            #______________OPTIMIZE ROTATION_______________________

            costR = lambda R: self.error(self.W, R, mean)
            problemR = Problem(manifold=V, cost=pymanopt.function.jax(V)(costR))
            solverR = SteepestDescent(max_iterations=10, verbosity=verbose)
            Rnew = solverR.run(problemR, initial_point=self.rotation).point

            changeInSS_rotation = jnp.max(jnp.sum(jnp.abs(Rnew-self.rotation), axis=1))

            #update self.rotation to the new rotation
            self.rotation = Rnew

            #check stopping criteria
            if it>0:
                maxChange = max([float(changeInSS_rotation), float(changeInSS_weights)])
                if epsilon <= tol * (1 + cost) and maxChange <= tol**.5:
                    if verbose:
                        print(f"Stopping criteria reached in iteration {it}.")
                    break

            #________________OPTIMIZE WEIGHTS_______________________

            E = Euclidean(self.weights.size)
            costC = lambda c: self.error(self.W, Rnew, self.mean(self.inputShapes, c))
            problemC = Problem(manifold=E, cost=pymanopt.function.jax(E)(costC))
            solverC = SteepestDescent(max_iterations=5, verbosity=0)
            Cnew = solverC.run(problemC, initial_point=self.weights).point

            changeInSS_weights = jnp.max(jnp.abs(Cnew - self.weights))
            self.weights = Cnew
            mean = self.mean(self.inputShapes, self.weights)

            #_________________________________________________________

            newCost = error(self.rotation, self.weights)
            if verbose:
                print("Cost: ", newCost)
            #_________________________________________________________

        #get the 3rd row of rotation matrix by cross multiplying first 2 rows
        rotation3d = jnp.zeros((3,3))
        rotation3d = rotation3d.at[:, :2].set(self.rotation[:, :2])
        rotation3d = rotation3d.at[:, 2].set(np.cross(*self.rotation.T))

        self.frechetMean = mean @ rotation3d
        return self.frechetMean
