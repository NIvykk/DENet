import importlib
import math
import os
import time
from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import utils as vutils

# Settings
# format short g, %precision=5
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})
# number of multiprocessing threads
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))
# prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
cv2.setNumThreads(0)
# NumExpr max threads
os.environ['NUMEXPR_MAX_THREADS'] = str(NUM_THREADS)
# OpenMP max threads (PyTorch and SciPy)
os.environ['OMP_NUM_THREADS'] = str(NUM_THREADS)


def time_synchronized(use_cpu=False):
    torch.cuda.synchronize(
    ) if torch.cuda.is_available() and not use_cpu else None
    return time.time()


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel,
                           nn.parallel.DistributedDataParallel)


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def profile(x, ops, n=100, device=None):
    # profile a pytorch module or list of modules. Example usage:
    #     x = torch.randn(16, 3, 640, 640)  # input
    #     m1 = lambda x: x * torch.sigmoid(x)
    #     m2 = nn.SiLU()
    #     profile(x, [m1, m2], n=100)  # profile speed over 100 iterations

    device = device or torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    x.requires_grad = True
    print(torch.__version__, device.type,
          torch.cuda.get_device_properties(0) if device.type == 'cuda' else '')
    print(
        f"\n{'Params':>12s}{'GFLOPS':>12s}{'forward (ms)':>16s}{'backward (ms)':>16s}{'input':>24s}{'output':>24s}"
    )
    for m in ops if isinstance(ops, list) else [ops]:
        m = m.to(device) if hasattr(m, 'to') else m  # device
        m = m.half() if hasattr(m, 'half') and isinstance(
            x, torch.Tensor) and x.dtype is torch.float16 else m  # type
        dtf, dtb, t = 0., 0., [0., 0., 0.]  # dt forward, backward
        try:
            flops = thop.profile(m, inputs=(x, ),
                                 verbose=False)[0] / 1E9 * 2  # GFLOPS
        except:
            flops = 0

        for _ in range(n):
            t[0] = time_synchronized()
            y = m(x)
            t[1] = time_synchronized()
            try:
                _ = y.sum().backward()
                t[2] = time_synchronized()
            except:  # no backward method
                t[2] = float('nan')
            dtf += (t[1] - t[0]) * 1000 / n  # ms per op forward
            dtb += (t[2] - t[1]) * 1000 / n  # ms per op backward

        s_in = tuple(x.shape) if isinstance(x, torch.Tensor) else 'list'
        s_out = tuple(y.shape) if isinstance(y, torch.Tensor) else 'list'
        p = sum(list(x.numel() for x in m.parameters())) if isinstance(
            m, nn.Module) else 0  # parameters
        print(
            f'{p:12.4g}{flops:12.4g}{dtf:16.4g}{dtb:16.4g}{str(s_in):>24s}{str(s_out):>24s}'
        )


def model_info(model, verbose=False):
    import thop

    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters()
              if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' %
              ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu',
               'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(
                      p.shape), p.mean(), p.std()))

    # try:  # FLOPS
    device = next(model.parameters()).device  # get model device
    flops = thop.profile(deepcopy(model.eval()),
                         inputs=(torch.zeros(1, 3, 640, 352).to(device), ),
                         verbose=False)[0] / 1E9 * 2
    fs = ', %.1f GFLOPS' % (flops)  # 640x352 FLOPS
    # except:
    #     fs = ''

    print('Model Summary: %g layers, %g parameters, %g gradients%s' %
          (len(list(model.parameters())), n_p, n_g, fs))


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(
            model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(
            -x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(
                model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    @staticmethod
    def copy_attr(a, b, include=(), exclude=()):
        # Copy attributes from b to a, options to only include [...] and to exclude [...]
        for k, v in b.__dict__.items():
            if (len(include)
                    and k not in include) or k.startswith('_') or k in exclude:
                continue
            else:
                setattr(a, k, v)

    def update_attr(self,
                    model,
                    include=('hyp', 'gr', 'num_classes', 'class_names'),
                    exclude=('process_group', 'reducer')):
        # Update EMA attributes
        self.copy_attr(self.ema, model, include, exclude)


def de_normalize(tensor,
                 mean=[0.5, 0.5, 0.5],
                 std=[0.5, 0.5, 0.5],
                 inplace=False):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            'Input tensor should be a torch tensor. Got {}.'.format(
                type(tensor)))

    if tensor.ndim < 3:
        raise ValueError(
            'Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
            '{}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(
            'std evaluated to zero after conversion to {}, leading to division by zero.'
            .format(dtype))
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)

    # tensor.sub_(mean).div_(std)  # normalize
    tensor.mul_(std).add_(mean)  # de_normalize

    return tensor


# from tyf
def import_fun(fun_dir, module):
    fun = module.split('.')
    m = importlib.import_module(fun_dir + '.' + fun[0])
    return getattr(m, fun[1])


def get_gpu_mem():
    mem = '%.3gG' % (torch.cuda.memory_reserved() /
                     1E9 if torch.cuda.is_available() else 0)  # (GB)
    return mem


def rgb2gray(im):
    assert isinstance(im, torch.Tensor)
    assert im.ndim == 4
    im_gray = im[:,
                 0, :, :] * 0.299 + im[:,
                                       1, :, :] * 0.587 + im[:,
                                                             2, :, :] * 0.114
    return im_gray.unsqueeze(1)


def save_tensor_to_image(input_tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.detach().cpu()
    vutils.save_image(input_tensor, filename)
