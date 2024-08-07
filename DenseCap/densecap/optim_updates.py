import torch

def sgd(x, dx, lr):
    x.add_(-lr, dx)

def sgdm(x, dx, lr, alpha, state):
    if 'v' not in state:
        state['v'] = torch.zeros_like(x)
    state['v'].mul_(alpha).add_(lr, dx)
    x.add_(-1, state['v'])

def sgdmom(x, dx, lr, alpha, state):
    if 'm' not in state:
        state['m'] = torch.zeros_like(x)
        state['tmp'] = torch.zeros_like(x)
    state['tmp'].copy_(state['m'])
    state['m'].mul_(alpha).add_(-lr, dx)
    x.add_(-alpha, state['tmp'])
    x.add_(1 + alpha, state['m'])

def adagrad(x, dx, lr, epsilon, state):
    if 'm' not in state:
        state['m'] = torch.zeros_like(x)
        state['tmp'] = torch.zeros_like(x)
    state['m'].addcmul_(1.0, dx, dx)
    state['tmp'].sqrt_(state['m']).add_(epsilon)
    x.addcdiv_(-lr, dx, state['tmp'])

def rmsprop(x, dx, lr, alpha, epsilon, state):
    if 'm' not in state:
        state['m'] = torch.zeros_like(x)
        state['tmp'] = torch.zeros_like(x)
    state['m'].mul_(alpha).addcmul_(1.0 - alpha, dx, dx)
    state['tmp'].sqrt_(state['m']).add_(epsilon)
    x.addcdiv_(-lr, dx, state['tmp'])

def adam(x, dx, lr, beta1, beta2, epsilon, state):
    beta1 = beta1 or 0.9
    beta2 = beta2 or 0.999
    epsilon = epsilon or 1e-8

    if 'm' not in state:
        state['t'] = 0
        state['m'] = torch.zeros_like(dx)
        state['v'] = torch.zeros_like(dx)
        state['tmp'] = torch.zeros_like(dx)

    state['m'].mul_(beta1).add_(1 - beta1, dx)
    state['v'].mul_(beta2).addcmul_(1 - beta2, dx, dx)
    state['tmp'].copy_(state['v']).sqrt_().add_(epsilon)

    state['t'] += 1
    bias_correction1 = 1 - beta1 ** state['t']
    bias_correction2 = 1 - beta2 ** state['t']
    step_size = lr * (bias_correction2 ** 0.5) / bias_correction1

    x.addcdiv_(-step_size, state['m'], state['tmp'])
