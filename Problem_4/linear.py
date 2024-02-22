import torch
import torch.nn as nn


class BoundLinear(nn.Linear):
    def __init(self, in_features, out_features, bias=True):
        super(BoundLinear, self).__init__(in_features, out_features, bias)

    @staticmethod
    def convert(linear_layer):
        r"""Convert a nn.Linear object into a BoundLinear object

        Args: 
            linear_layer (nn.Linear): The linear layer to be converted.

        Returns:
            l (BoundLinear): The converted layer
        """
        l = BoundLinear(linear_layer.in_features, linear_layer.out_features, linear_layer.bias is not None)
        l.weight.data.copy_(linear_layer.weight.data)
        l.weight.data = l.weight.data.to(linear_layer.weight.device)
        l.bias.data.copy_(linear_layer.bias.data)
        l.bias.data = l.bias.to(linear_layer.bias.device)
        return l

    def boundpropogate(self, last_uA, last_lA, start_node=None):
        r"""Bound propagate through the linear layer

        Args:
            last_uA (tensor): A (the coefficient matrix) that is bound-propagated to this layer
            (from the layers after this layer). It's exclusive for computing the upper bound.

            last_lA (tensor): A that is bound-propagated to this layer. It's exclusive for computing the lower bound.

            start_node (int): An integer indicating the start node of this bound propagation. (It's not used in linear layer)


        Returns:
            uA (tensor): The new A for computing the upper bound after taking this layer into account.

            ubias (tensor): The bias (for upper bound) produced by this layer.

            lA( tensor): The new A for computing the lower bound after taking this layer into account.

            lbias (tensor): The bias (for lower bound) produced by this layer.
        """

        def _bound_oneside(last_A):
            if last_A is None:
                return None, 0
            # propagate A to the nest layer
            next_A = last_A.matmul(self.weight)
            # compute the bias of this layer
            sum_bias = last_A.matmul(self.bias)
            return next_A, sum_bias

        uA, ubias = _bound_oneside(last_uA)
        lA, lbias = _bound_oneside(last_lA)
        return uA, ubias, lA, lbias
