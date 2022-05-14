import pdb
import torch

class BKIConvFilter(torch.autograd.Function):
    """
    Custom autograd implementation for Bayesian Kernel Inference Filter

    Applies Sigmoid to all non center entries, keeping center entry 1
    """

    @staticmethod
    def forward(ctx, weights, mid):
        """
        Applies sigmoid to input weights, making sure to set center of 3D filter to 1
        to preserve bayesian kernel assumption

        @param weights  - B x C x X x Y Z 
        """
        ctx.save_for_backward(weights, mid)
        filters = torch.sigmoid(weights)
        filters[0, 0, mid, mid, mid] = 1
        return filters

    @staticmethod
    def backward(ctx, upstream_grad):
        """
        Gradients for d o(x) = o(x)*( 1-o(x) ), we dont want to update weight for
        center weight, so we simply zero since it has zero contribution
        """
        weights, mid = ctx.saved_tensors
        gw = torch.sigmoid(weights) * (1 - torch.sigmoid(weights))
        gw[0, 0, mid, mid, mid] = 0
        gm = None
        return upstream_grad * gw, gm