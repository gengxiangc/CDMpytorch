# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 17:36:39 2019

Y_new = Ys * w(x) + b(x)
Conditional distribution matching by min P(Ys_new|Xs) - P(Yt|Xt)

@author: 陈耿祥
"""
import torch 
from torch.autograd import Variable
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

''' Example 1'''
if 0:
    loss = loss=torch.nn.MSELoss()
    n = 100
    x = torch.rand((n))
    y = x*3 + 1 + torch.rand(n)/5
    k = Variable(torch.tensor([1]), requires_grad=True)
    b = Variable(torch.tensor([0]), requires_grad=True) 
    f = k*x+b
    LR = 0.3
    opt_SGD = torch.optim.SGD([k, b], lr=LR)
    for epoch in range(100):
        l = loss(f, y)
        opt_SGD.zero_grad()
        l.backward()
        opt_SGD.step()
        print("k={:.2},b={:.2},l={:.2}".format(k.data[0],b.data[0],l.data))
    
       
def pdinv(A):
    n = len(A)
    U = scipy.linalg.cholesky(A)
    invU = torch.ones(n)/U
    Ainv = torch.inverse(torch.mm(invU, invU.t()))
    return Ainv
    
def kernel(ker, X, X2, sigma):
    '''
    Pytorch
    Input: X  n_feature*Size1
           X2 n_feature*Size2
    Output: Size1*Size2
    '''
    n1, n2 = X.shape[1],X2.shape[1]
    if  ker == 'linear':
        K = torch.mm(X.t(), X2)
    elif ker == 'rbf':
        n1sq = torch.sum(X ** 2, 0)        
        n2sq = torch.sum(X2 ** 2, 0)
        D = torch.ones((n1, n2), dtype = torch.double).mul(n2sq) +  \
            torch.ones((n2, n1), dtype = torch.double).mul(n1sq).t() - \
            2 * torch.mm(X.t(), X2)
        K = torch.exp(-sigma * D)
    elif ker == 'sam':
        D = X.t().mm(X2)
        K = torch.exp(-sigma * torch.acos(D) ** 2)
    return K

#class LS_minPYX():
def train (Xs, Xt, Ys, Yt, sigma, 
    lambda_regularization = 1e-3, 
    lambda_inv = 0.1, 
    learning_rate = 0.9,
    Max_Iter = 100,
    Thresh = 1e-5):
    """
    Y_new = Ys * w(x) + b(x)
    Conditional distribution matching by min P(Ys_new|Xs) - P(Yt|Xt)
    Parameters
    ----------
    Xs : X of source domain
    Xt : X of target domain
    Ys : X of source domain
    Yt : X of target domain
    sigma: int 
        the kernel width for Y used to construct Gram matrix K
    """

    # Initial parameters
    wide_kernel = sigma*2
    lambda_inv  = 0.1
    Tol         = 1e-6
    Max_Iter    = Max_Iter
    LR          = learning_rate
    ns, nt      = len(Xs), len(Xt)
    
    # to torch 
    Xs = torch.from_numpy(Xs)
    Xt = torch.from_numpy(Xt)
    Ys = torch.from_numpy(Ys)
    Yt = torch.from_numpy(Yt)


    # Kernel matrix [constant]
    KXs     = kernel('rbf', Xs.t(), Xs.t(), wide_kernel)
    KXt     = kernel('rbf', Xt.t(), Xt.t(), wide_kernel)
    KXs_inv = torch.inverse(KXs + lambda_inv*torch.eye(ns, dtype = torch.double))
    KXt_inv = torch.inverse(KXt + lambda_inv*torch.eye(nt, dtype = torch.double))    
    KXtXs   = kernel('rbf', Xt.t(), Xs.t(), wide_kernel)

    # Find R [constant]
    e, V = torch.eig(KXs.mm(KXs_inv),eigenvectors=True)
    mask = e[:,0].gt(torch.max(e[:,0] * Thresh))
    R    = KXs.mm(KXs_inv).mm(V[mask].t()) # ns * n_egenvectors

    # initial params0 [constant]   
    temp0    = torch.inverse(R.t().mm(R)).mm(R.t()).mm(torch.ones((ns, 1),dtype = torch.double))
    params_W = torch.reshape(temp0, (R.shape[1], 1))
    params_B = torch.zeros((R.shape[1], 1), dtype = torch.double)
    
    # Set variable grads
    params_W = Variable(params_W, requires_grad=True)
    params_B = Variable(params_B, requires_grad=True)

    # Begin to optimize params
    Error      = 1
    Iteriation = 0
    
    # loss function 
    opt_SGD = torch.optim.SGD([params_W, params_B], lr=LR)
    while (Error > Tol) & (Iteriation < Max_Iter):
        Iteriation+=1
        W        = R.mm(params_W)
        B        = R.mm(params_B)
        Ys_new   = Ys.mul(W) + B
        tilde_K  = kernel('rbf', Ys_new.t(), Ys_new.t(), wide_kernel)
        tilde_Kc = kernel('rbf', Yt.t(), Ys_new.t(), wide_kernel)
        part1    = torch.trace(KXs_inv.mm(tilde_K).mm(KXs_inv).mm(KXs))
        part2    = 2 * torch.trace(KXs_inv.mm(tilde_Kc.t()).mm(KXt_inv).mm(KXtXs))
        W_       = W-torch.ones(W.shape,dtype = torch.double)
        part3    = lambda_regularization*\
                     (torch.sum(W_.mul(W_)) + torch.sum(B.mul(B)))
        loss     = part1 - part2 + part3
        opt_SGD.zero_grad()
        loss.backward()
        opt_SGD.step()
        print("loss={:.2}".format(loss))
        
    return Ys_new

if __name__ == '__main__':    
    
    sigma    = 0.5
    lbd_reg  = 1e-4
    lbd_inv  = 0.1
    lng_rate = 0.1
    max_iter = 100
    Thresh   = 1e-5
    
    Title = 'Parameters: '+'sigma='+str(sigma)+\
            ' lambdaInv='+str(lbd_inv)+' LR='+str(lng_rate)
            
    # Demon curve 3
    if 1:  
        Xs_ = np.linspace(-5, 5, 126)
        Ys = np.sin(Xs_) + 1
        Ys = Ys[:, np.newaxis]
        
        Xt_ = np.linspace(-5, 2, 3)
        Yt = 0.6*np.sin(Xt_) + 0.5
        Yt = Yt[:, np.newaxis]
                  
        Xs = np.vstack((Xs_, Xs_)).T
        Xt = np.vstack((Xt_, Xt_)).T
        
        Xtest = Xs
        
    
    # main
    Ys_new = train(Xs, Xt, Ys, Yt, sigma = sigma,                    
                   lambda_regularization = lbd_reg, 
                   lambda_inv = lbd_inv, 
                   learning_rate = lng_rate,
                   Max_Iter = max_iter,
                   Thresh = Thresh)
       
    # fig
    Ys_new = Ys_new.detach().numpy()
    fig, ax = plt.subplots()    
    plt.plot(Xs_, Ys_new, 'purple', lw=2, zorder=9, label='Min P(Y|X)')     
    plt.plot(Xs_, Ys, 'r-', lw=3, label='Source model')
    plt.scatter(Xt_, Yt, c='b', s=100, label='Target Data')
    fontfamily = 'NSimSun'
    font = {'family':fontfamily,
            'size':12,
            'weight':23}   
    ax.set_xlabel('X',fontproperties = fontfamily, size = 12)
    ax.set_ylabel('Y',fontproperties = fontfamily, size = 12)
    plt.yticks(fontproperties = fontfamily, size = 12) 
    plt.xticks(fontproperties = fontfamily, size = 12) 
    ax.set_title(Title, fontproperties = fontfamily, size = 12)
    plt.legend(prop=font)
    plt.tight_layout()
    plt.ylim([-2,3])
    plt.legend(prop=font)
    plt.show()


        






        


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    