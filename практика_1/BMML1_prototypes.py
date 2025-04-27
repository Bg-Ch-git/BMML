# There should be no main() in this file!!!
# Nothing should start running when you import this file somewhere.
# You may add other supporting functions to this file.
#
# Important rules:
# 1) Function pa_bc must return tensor which has dimensions (#a x #b x #c),
#    where #v is a number of different values of the variable v.
#    For input variables #v = how many input values of this variable you give to the function.
#    For output variables #v = number of all possible values of this variable.
#    Ex. for pb_a: #b = bmax-bmin+1,   #a is arbitrary.
# 2) Random variables in function names must be written in alphabetic order
#    e.g. pda_cb is an improper function name (pad_bc must be used instead)
# 3) Single dimension must be explicitly stated:
#    if you give only one value of a variable a to the function pb_a, i.e. #a=1,
#    then the function pb_a must return tensor of shape (#b, 1), not (#b,).
#
# The format of all the functions for distributions is the following:
# Inputs:
# params - dictionary with keys 'amin', 'amax', 'bmin', 'bmax', 'p1', 'p2', 'p3'
# model - model number, number from 1 to 4
# all other parameters - values of the conditions (variables a, b, c, d).
#                        Numpy vectors of size (k,), where k is an arbitrary number.
#                        For variant 3: c and d must be numpy arrays of size (k,N),
#                        where N is a number of lectures.
# Outputs:
# prob, val
# prob - probabilities for different values of the output variable with different input conditions
#        prob[i,...] = p(v=val[i]|...)
# val - support of a distribution, numpy vector of size (#v,) for variable v
#
# Example 1:
#    Function pc_ab - distribution p(c|a,b)
#    Input: a of size (k_a,) and b of size (k_b,)
#    Result: prob of size (cmax-cmin+1,k_a,k_b), val of size (cmax-cmin+1,)
#
# Example 2 (for variant 3):
#    Function pb_ad - distribution p(b|a,d_1,...,d_N)
#    Input: a of size (k_a,) and d of size (k_d,N)
#    Result: prob of size (bmax-bmin+1,k_a,k_d), val of size (bmax-bmin+1,)
#
# The format the generation function from variant 3 is the following:
# Inputs:
# N - how many points to generate
# all other inputs have the same format as earlier
# Outputs:
# d - generated values of d, numpy array of size (N,#a,#b)

import numpy as np
from scipy.stats import poisson, binom

# In variant 2 the following functions are required:


def pa(params, model):
    num = params['amax'] - params['amin'] + 1
    return np.ones(num) / num, np.arange(params['amin'], params['amax'] + 1)


def pb(params, model):
    num = params['bmax'] - params['bmin'] + 1
    return np.ones(num) / num, np.arange(params['bmin'], params['bmax'] + 1)


def pc(params, model):
    clen = params['amax'] + params['bmax'] + 1
    res_prob = np.zeros(clen)
    res_val = np.arange(clen)
    if model == 2:
        for i in range(clen):
            a = np.arange(params['amin'], params['amax'] + 1)
            b = np.arange(params['bmin'], params['bmax'] + 1)
            res_prob[i] = poisson.pmf(res_val[i], np.expand_dims(
                a, axis=0).T * params['p1'] + b * params['p2']).sum()
    else:
        for a in range(params['amin'], params['amax'] + 1):
            for b in range(params['bmin'], params['bmax'] + 1):
                res_prob += np.convolve(binom.pmf(res_val, a, params['p1']), binom.pmf(
                    res_val, b, params['p2']))[:res_prob.shape[0]]
    return res_prob / ((params['amax'] - params['amin'] + 1) * (params['bmax'] - params['bmin'] + 1)), res_val


def pd(params, model):
    prob_c, val_c = pc(params, model)
    res_prob = np.zeros(2 * (params['amax'] + params['bmax']) + 1)
    res_val = np.arange(2 * (params['amax'] + params['bmax']) + 1)
    for c in val_c:
        res_prob += prob_c[c] * binom.pmf(res_val, c, params['p3'])
    return res_prob, res_val


def pc_a(a, params, model):
    clen = params['amax'] + params['bmax'] + 1
    res_prob = np.zeros((clen, a.shape[0]))
    res_val = np.arange(clen)
    if model == 2:
        for i in range(clen):
            b = np.arange(params['bmin'], params['bmax'] + 1)
            res_prob[i, :] = poisson.pmf(res_val[i], np.expand_dims(
                a, axis=0).T * params['p1'] + b * params['p2']).sum(axis=1)
    else:
        for j in range(a.shape[0]):
            for b in range(params['bmin'], params['bmax'] + 1):
                res_prob[:, j] += np.convolve(binom.pmf(res_val, a[j], params['p1']), binom.pmf(
                    res_val, b, params['p2']))[:res_prob.shape[0]]
    return res_prob / (params['bmax'] - params['bmin'] + 1), res_val


def pc_b(b, params, model):
    clen = params['amax'] + params['bmax'] + 1
    res_prob = np.zeros((clen, b.shape[0]))
    res_val = np.arange(clen)
    if model == 2:
        for i in range(clen):
            a = np.arange(params['amin'], params['amax'] + 1)
            res_prob[i, :] = poisson.pmf(res_val[i], np.expand_dims(
                b, axis=0).T * params['p2'] + a * params['p1']).sum(axis=1)
    else:
        for j in range(b.shape[0]):
            for a in range(params['amin'], params['amax'] + 1):
                res_prob[:, j] += np.convolve(binom.pmf(res_val, b[j], params['p2']), binom.pmf(
                    res_val, a, params['p1']))[:res_prob.shape[0]]
    return res_prob / (params['amax'] - params['amin'] + 1), res_val


def pb_a(a, params, model):
    num = params['bmax'] - params['bmin'] + 1
    return np.ones((num, a.shape[0])) / num, np.arange(params['bmin'], params['bmax'] + 1)


def pb_d(d, params, model):
    blen = params['bmax'] - params['bmin'] + 1
    res_prob = np.zeros((blen, d.shape[0]))
    res_val = np.arange(params['bmin'], params['bmax'] + 1)
    prob_pc_b, val_c = pc_b(res_val, params, model)
    for i in range(d.shape[0]):
        res_prob[:, i] = binom.pmf(
            d[i] - val_c, val_c, params['p3']).dot(prob_pc_b)
    return res_prob / res_prob.sum(axis=0), res_val


def pb_ad(a, d, params, model):
    blen = params['bmax'] - params['bmin'] + 1
    alen = params['amax'] - params['amin'] + 1
    clen = params['amax'] + params['bmax'] + 1
    c_val = np.arange(clen)
    res_prob = np.zeros((blen, a.shape[0], d.shape[0]))
    res_val = np.arange(params['bmin'], params['bmax'] + 1)
    b = res_val
    if model == 2:
        for i in range(blen):
            for j in range(a.shape[0]):
                for k in range(d.shape[0]):
                    res_prob[i, j, k] += binom.pmf(d[k] - c_val, c_val, params['p3']).dot(
                        poisson.pmf(c_val, a[j] * params['p1'] + b[i] * params['p2']))
    else:
        for i in range(blen):
            for j in range(a.shape[0]):
                for k in range(d.shape[0]):
                    res_prob[i, j, k] += binom.pmf(d[k] - c_val, c_val, params['p3']).dot(np.convolve(binom.pmf(c_val, b[i], params['p2']), binom.pmf(
                        c_val, a[j], params['p1']))[:c_val.shape[0]])
    return res_prob / res_prob.sum(axis=0), res_val
