import torch
import torch.nn as nn
import numpy as np
from model import SimpleNNRelu, SimpleNNHardTanh
from linear import BoundLinear
from relu import BoundReLU
from hardTanh_question import BoundHardTanh
import time
import argparse


class BoundedSequential(nn.Sequential):
    r"""This class wraps the PyTorch nn.Sequential object with bound computation."""

    def __init__(self, *args):
        super(BoundedSequential, self).__init__(*args)

    @staticmethod
    def convert(seq_model):
        r"""Convert a Pytorch model to a model with bounds.
        Args:
            seq_model: An nn.Sequential module.

        Returns:
            The converted BoundedSequential module.
        """
        layers = []
        for l in seq_model:
            if isinstance(l, nn.Linear):
                layers.append(BoundLinear.convert(l))
            elif isinstance(l, nn.ReLU):
                layers.append(BoundReLU.convert(l))
            elif isinstance(l, nn.Hardtanh):
                layers.append(BoundHardTanh.convert(l))
        return BoundedSequential(*layers)

    def compute_bounds(self, x_U=None, x_L=None, upper=True, lower=True, optimize=False):
        r"""Main function for computing bounds.

        Args:
            x_U (tensor): The upper bound of x.

            x_L (tensor): The lower bound of x.

            upper (bool): Whether we want upper bound.

            lower (bool): Whether we want lower bound.

            optimize (bool): Whether we optimize alpha.

        Returns:
            ub (tensor): The upper bound of the final output.

            lb (tensor): The lower bound of the final output.
        """
        ub = lb = None
        ub, lb = self.full_boundpropogation(x_U=x_U, x_L=x_L, upper=upper, lower=lower)
        return ub, lb

    def full_boundpropogation(self, x_U=None, x_L=None, upper=True, lower=True):
        r"""A full bound propagation. We are going to sequentially compute the
        intermediate bounds for each linear layer followed by a ReLU layer. For each
        intermediate bound, we call self.boundpropogate_from_layer() to do a bound propagation
        starting from that layer.

        Args:
            x_U (tensor): The upper bound of x.

            x_L (tensor): The lower bound of x.

            upper (bool): Whether we want upper bound.

            lower (bool): Whether we want lower bound.

        Returns:
            ub (tensor): The upper bound of the final output.

            lb (tensor): The lower bound of the final output.
        """
        modules = list(self._modules.values())
        # CROWN propagation for all layers
        for i in range(len(modules)):
            # We only need the bounds before a ReLU/HardTanh layer
            if isinstance(modules[i], BoundReLU) or isinstance(modules[i], BoundHardTanh):
                if isinstance(modules[i - 1], BoundLinear):
                    # add a batch dimension
                    newC = torch.eye(modules[i - 1].out_features).unsqueeze(0).repeat(x_U.shape[0], 1, 1).to(x_U)
                    # Use CROWN to compute pre-activation bounds
                    # starting from layer i-1
                    ub, lb = self.boundpropogate_from_layer(x_U=x_U, x_L=x_L, C=newC, upper=True, lower=True,
                                                            start_node=i - 1)
                # Set pre-activation bounds for layer i (the ReLU layer)
                modules[i].upper_u = ub
                modules[i].lower_l = lb
        # Get the final layer bound
        return self.boundpropogate_from_layer(x_U=x_U, x_L=x_L,
                                              C=torch.eye(modules[i].out_features).unsqueeze(0).to(x_U), upper=upper,
                                              lower=lower, start_node=i)

    def boundpropogate_from_layer(self, x_U=None, x_L=None, C=None, upper=False, lower=True, start_node=None):
        r"""The bound propagation starting from a given layer. Can be used to compute intermediate bounds or the final bound.

        Args:
            x_U (tensor): The upper bound of x.

            x_L (tensor): The lower bound of x.

            C (tensor): The initial coefficient matrix. Can be used to represent the output constraints.
            But we don't have any constraints here. So it's just an identity matrix.

            upper (bool): Whether we want upper bound.

            lower (bool): Whether we want lower bound.

            start_node (int): The start node of this propagation. It should be a linear layer.
        Returns:
            ub (tensor): The upper bound of the output of start_node.
            lb (tensor): The lower bound of the output of start_node.
        """
        modules = list(self._modules.values()) if start_node is None else list(self._modules.values())[:start_node + 1]
        upper_A = C if upper else None
        lower_A = C if lower else None
        upper_sum_b = lower_sum_b = x_U.new([0])
        for i, module in enumerate(reversed(modules)):
            upper_A, upper_b, lower_A, lower_b = module.boundpropogate(upper_A, lower_A, start_node)
            upper_sum_b = upper_b + upper_sum_b
            lower_sum_b = lower_b + lower_sum_b

        # sign = +1: upper bound, sign = -1: lower bound
        def _get_concrete_bound(A, sum_b, sign=-1):
            if A is None:
                return None
            A = A.view(A.size(0), A.size(1), -1)
            # A has shape (batch, specification_size, flattened_input_size)
            x_ub = x_U.view(x_U.size(0), -1, 1)
            x_lb = x_L.view(x_L.size(0), -1, 1)
            center = (x_ub + x_lb) / 2.0
            diff = (x_ub - x_lb) / 2.0
            bound = A.bmm(center) + sign * A.abs().bmm(diff)
            bound = bound.squeeze(-1) + sum_b
            return bound

        lb = _get_concrete_bound(lower_A, lower_sum_b, sign=-1)
        ub = _get_concrete_bound(upper_A, upper_sum_b, sign=+1)
        if ub is None:
            ub = x_U.new([np.inf])
        if lb is None:
            lb = x_L.new([-np.inf])
        return ub, lb


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--activation', default='relu', choices=['relu', 'hardtanh'],
                        type=str, help='Activation Function')
    parser.add_argument('data_file', type=str, help='input data, a tensor saved as a .pth file.')
    # Parse the command line arguments
    args = parser.parse_args()

    x_test, label = torch.load(args.data_file)

    if args.activation == 'relu':
        print('use ReLU model')
        model = SimpleNNRelu()
        model.load_state_dict(torch.load('models/relu_model.pth'))
    else:
        print('use HardTanh model')
        model = SimpleNNHardTanh()
        model.load_state_dict(torch.load('models/hardtanh_model.pth'))

    batch_size = x_test.size(0)
    x_test = x_test.reshape(batch_size, -1)
    output = model(x_test)
    y_size = output.size(1)
    print("Network prediction: {}".format(output))

    eps = 0.01
    x_u = x_test + eps
    x_l = x_test - eps

    print(f"Verifiying Pertubation - {eps}")
    start_time = time.time()
    boundedmodel = BoundedSequential.convert(model)
    ub, lb = boundedmodel.compute_bounds(x_U=x_u, x_L=x_l, upper=True, lower=True)
    for i in range(batch_size):
        for j in range(y_size):
            print('f_{j}(x_{i}): {l:8.4f} <= f_{j}(x_{i}+delta) <= {u:8.4f}'.format(
                j=j, i=i, l=lb[i][j].item(), u=ub[i][j].item()))

