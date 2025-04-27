import numpy as np
from scipy.stats import norm
from scipy.signal import fftconvolve

def calculate_log_probability(X, F, B, s):
    """
    Calculates log p(X_k|d_k,F,B,s) for all images X_k in X and
    all possible displacements d_k.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.

    Returns
    -------
    ll : array, shape(H-h+1, W-w+1, K)
        ll[dh,dw,k] - log-likelihood of observing image X_k given
        that the villain's face F is located at displacement (dh, dw)
    """
    helper_kernel = np.ones_like(F)
    result = np.zeros((X.shape[0] - F.shape[0] + 1, X.shape[1] - F.shape[1] + 1, X.shape[2]))
    for_B = (B * B).sum() - fftconvolve(B * B, np.rot90(helper_kernel, k=2), mode='valid')
    for_B /= 2 * s * s
    for k in range(X.shape[-1]):
        result[..., k] -= (F * F).sum() / (2 * s * s) + (X[..., k] ** 2).sum() / (2 * s * s) + (X.shape[0] * X.shape[1] * np.log(2 * np.pi * s * s)) / 2 + for_B
        result[..., k] += ((X[..., k] * B).sum() - fftconvolve(X[..., k] * B, np.rot90(helper_kernel, k=2), mode='valid') + fftconvolve(X[..., k], np.rot90(F, k=2), mode='valid')) / (s * s)
    return result


def calculate_lower_bound(X, F, B, s, A, q, use_MAP=False):
    """
    Calculates the lower bound L(q,F,B,s,A) for the marginal log likelihood.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    L : float
        The lower bound L(q,F,B,s,A) for the marginal log likelihood.
    """
    log_p_X_d = calculate_log_probability(X, F, B, s)
    res = np.sum(-q[q!=0] * np.log(q[q!=0]))
    A[A <= 1e-308] = 1e-308
    helper_A = log_p_X_d + np.log(A)[..., None]
    res += np.sum(helper_A[q != 0] * q[q != 0])
    return res


def run_e_step(X, F, B, s, A, use_MAP=False):
    """
    Given the current esitmate of the parameters, for each image Xk
    esitmates the probability p(d_k|X_k,F,B,s,A).

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    F  : array_like, shape(h, w)
        Estimate of villain's face.
    B : array shape(H, W)
        Estimate of background.
    s : scalar, shape(1, 1)
        Eestimate of standard deviation of Gaussian noise.
    A : array, shape(H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    use_MAP : bool, optional,
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    """
    enum_log = calculate_log_probability(X, F, B, s)
    maximums = np.max(enum_log, axis=(0,1))
    enum_log -= maximums[None, None, ...]
    enum = np.exp(enum_log) * A[..., None]
    sums = np.sum(enum, axis=(0,1))
    res = enum / sums[None, None, ...]
    return res
    maxs = np.max(res, axis=(0,1))
    helper_res = res / maxs[None,None, ...]
    res[helper_res < 0.01] = 0
    return res / np.sum(res, axis=(0,1))[None, None, ...]


def run_m_step(X, q, h, w, use_MAP=False):
    """
    Estimates F,B,s,A given esitmate of posteriors defined by q.

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    q  :
        if use_MAP = False: array, shape (H-h+1, W-w+1, K)
           q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
           of villain's face given image Xk
        if use_MAP = True: array, shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    h : int
        Face mask height.
    w : int
        Face mask width.
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    """
    A = np.mean(q, axis=-1)

    F = np.zeros((h, w))
    B = np.zeros((X.shape[0], X.shape[1]))
    helper_add = np.zeros_like(B)
    K = X.shape[-1]
    s = 0
    for k in range(K):
        F += fftconvolve(X[..., k], q[..., k], mode='valid')
        helper_prod = 1 - fftconvolve(q[..., k], np.rot90(np.ones((h, w)), k=2), mode='full')
        helper_prod[helper_prod < 0] = 0
        helper_prod[helper_prod > 1] = 1
        helper_add += helper_prod
        B += helper_prod * X[..., k]
    F /= K
    B[helper_add!=0] /= helper_add[helper_add!=0]
    B[helper_add==0] = X.mean(axis=-1)[helper_add==0]

    helper_kernel = np.ones_like(F)
    for_B = (B * B).sum() - fftconvolve(B * B, np.rot90(helper_kernel, k=2), mode='valid')
    for k in range(X.shape[-1]):
        res = (F * F).sum() + (X[..., k] ** 2).sum() + for_B
        res -= 2* ((X[..., k] * B).sum() - fftconvolve(X[..., k] * B, np.rot90(helper_kernel, k=2), mode='valid') + fftconvolve(X[..., k], np.rot90(F, k=2), mode='valid'))
        s += np.sum(q[..., k] * res)
    s /= K * X.shape[0] * X.shape[1]
    s = np.sqrt(s)
    return F, B, s, A


def run_EM(X, h, w, F=None, B=None, s=None, A=None, tolerance=0.001,
           max_iter=50, use_MAP=False):
    """
    Runs EM loop until the likelihood of observing X given current
    estimate of parameters is idempotent as defined by a fixed
    tolerance.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    F : array, shape (h, w), optional
        Initial estimate of villain's face.
    B : array, shape (H, W), optional
        Initial estimate of background.
    s : float, optional
        Initial estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1), optional
        Initial estimate of prior on displacement of face in any image.
    tolerance : float, optional
        Parameter for stopping criterion.
    max_iter  : int, optional
        Maximum number of iterations.
    use_MAP : bool, optional
        If true then after E-step we take only MAP estimates of displacement
        (dh,dw) of villain's face given image Xk.

    Returns
    -------
    F, B, s, A : trained parameters.
    LL : array, shape(number_of_iters,)
        L(q,F,B,s,A) after each EM iteration (1 iteration = 1 e-step + 1 m-step);
        number_of_iters is actual number of iterations that was done.
    """
    LL = []
    iter = 0
    diff = 1
    if s is None:
        s = 50
    if B is None:
        B = X.mean(axis=-1)
        cv2_imshow(B)
    if F is None:
        F = np.random.random((h, w)) * 350 # np.ones((h, w)) * 100
        cv2_imshow(F)
    if A is None:
        A = np.random.random((X.shape[0] - h + 1, X.shape[1] - w + 1))
        A /= np.sum(A)
    while (diff >= tolerance or iter == 0) and iter < max_iter:
        iter += 1
        q = run_e_step(X, F, B, s, A, use_MAP=False)
        F, B, s, A = run_m_step(X, q, h, w, use_MAP=False)
        cv2_imshow(B)
        cv2_imshow(F)
        print(s)
        if len(LL) == 0:
            LL.append(calculate_lower_bound(X, F, B, s, A, q, use_MAP=False))
        else:
            LL.append(calculate_lower_bound(X, F, B, s, A, q, use_MAP=False))
            diff = LL[-1] - LL[-2]
    return F, B, s, A, LL


def run_EM_with_restarts(X, h, w, tolerance=0.001, max_iter=50, use_MAP=False,
                         n_restarts=10):
    """
    Restarts EM several times from different random initializations
    and stores the best estimate of the parameters as measured by
    the L(q,F,B,s,A).

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    tolerance, max_iter, use_MAP : optional parameters for EM.
    n_restarts : int
        Number of EM runs.

    Returns
    -------
    F : array,  shape (h, w)
        The best estimate of villain's face.
    B : array, shape (H, W)
        The best estimate of background.
    s : float
        The best estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        The best estimate of prior on displacement of face in any image.
    L : float
        The best L(q,F,B,s,A).
    """
    pass
