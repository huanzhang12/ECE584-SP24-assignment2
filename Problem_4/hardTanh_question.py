import torch
import torch.nn as nn


class BoundHardTanh(nn.Hardtanh):
    def __init__(self):
        super(BoundHardTanh, self).__init__()

    @staticmethod
    def convert(act_layer):
        r"""Convert a HardTanh layer to BoundHardTanh layer

        Args:
            act_layer (nn.HardTanh): The HardTanh layer object to be converted.

        Returns:
            l (BoundHardTanh): The converted layer object.
        """
        # TODO: Return the converted HardTanH
        pass

    def boundpropogate(self, last_uA, last_lA, start_node=None):
        """
        Propagate upper and lower linear bounds through the HardTanh activation function
        based on pre-activation bounds.

        Args:
            last_uA (tensor): A (the coefficient matrix) that is bound-propagated to this layer
            (from the layers after this layer). It's exclusive for computing the upper bound.

            last_lA (tensor): A that is bound-propagated to this layer. It's exclusive for computing the lower bound.

            start_node (int): An integer indicating the start node of this bound propagation

        Returns:
            uA (tensor): The new A for computing the upper bound after taking this layer into account.

            ubias (tensor): The bias (for upper bound) produced by this layer.

            lA( tensor): The new A for computing the lower bound after taking this layer into account.

            lbias (tensor): The bias (for lower bound) produced by this layer.

        """
        # These are preactivation bounds that will be used for form the linear relaxation.
        preact_lb = self.lower_l
        preact_ub = self.upper_u

        """
         Hints: 
         1. Have a look at the section 3.2 of the CROWN paper [1] (Case Studies) as to how segments are made for multiple activation functions
         2. Look at the HardTanH graph, and see multiple places where the pre activation bounds could be located
         3. Refer the ReLu example in the class and the diagonals to compute the slopes/intercepts
         4. The paper talks about 3 segments S+, S- and S+- for sigmoid and tanh. You should figure your own segments based on preactivation bounds for hardtanh.
         [1] https://arxiv.org/pdf/1811.00866.pdf
        """

        # You should return the linear lower and upper bounds after propagating through this layer.
        # Upper bound: uA is the coefficients, ubias is the bias.
        # Lower bound: lA is the coefficients, lbias is the bias.

        return uA, ubias, lA, lbias

