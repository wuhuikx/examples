import torch

def to_dense(optim):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor) and param.layout is torch._mkldnn:
            param.data = param.data.to_dense()
            if param._grad is not None:
                param._grad.data = param._grad.data.to_dense()
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor) and subparam.layout is torch._mkldnn:
                    subparam.data = subparam.data.to_dense()
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to_dense()

def to_mkldnn(optim):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor) and param.dtype == torch.float32:
             param.data = param.data.to_mkldnn()
             if param._grad is not None:
                 param._grad.data = param._grad.data.to_mkldnn()
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor) and subparam.dtype == torch.float32:
                    subparam.data = subparam.data.to_mkldnn()
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to_mkldnn()
