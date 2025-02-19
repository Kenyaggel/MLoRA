import torch
import torch.nn as nn
import torch.nn.functional as F


class MLoRAFCN(nn.Module):
    """
    A PyTorch replication of your TensorFlow MLoRAFCN layer.

    Args:
        n_domain (int): number of domains
        units (int): output dimension of this fully-connected layer
        in_dim (int): the input feature dimension
        activation (str or callable): activation function (e.g. 'relu').
        use_bias (bool): whether to use bias in the main MLP path
        lora_r (int): rank for LoRA
        lora_reduce (int): if > 0, used to override lora_r = max(units / lora_reduce, 1)
        dropout_rate (float): dropout probability for the main MLP path
        is_finetune (bool): placeholder indicating if we are in finetune mode
    """

    def __init__(self,
                 n_domain,
                 units,
                 in_dim=None,  # must be provided or inferred
                 activation=None,
                 use_bias=True,
                 kernel_initializer=nn.init.xavier_uniform_,
                 bias_initializer=nn.init.zeros_,
                 lora_r=4,
                 lora_reduce=-1,
                 dropout_rate=0.5,
                 is_finetune=False):
        super(MLoRAFCN, self).__init__()
        self.n_domain = n_domain
        self.units = units
        self.in_dim = in_dim
        self.activation_fn = self._get_activation(activation)
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.is_finetune = is_finetune

        # Possibly override lora_r based on lora_reduce
        if lora_r < 1 and lora_reduce >= 1:
            self.lora_r = max(int(units / lora_reduce), 1)
        else:
            self.lora_r = lora_r

        # Main MLP weights
        # W: [in_dim, units]
        self.kernel = nn.Parameter(torch.empty(in_dim, units))
        kernel_initializer(self.kernel)  # e.g. xavier init
        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(units))
            bias_initializer(self.bias)  # e.g. zeros
        else:
            self.bias = None

        # Domain-specific LoRA:
        # A: [n_domain, in_dim, lora_r]
        self.a_kernel = nn.Parameter(torch.empty(n_domain, in_dim, self.lora_r))
        # B: [n_domain, lora_r, units]
        self.b_kernel = nn.Parameter(torch.empty(n_domain, self.lora_r, units))

        # domain_bias: [n_domain, units]
        if self.use_bias:
            self.domain_bias = nn.Parameter(torch.empty(n_domain, units))
        else:
            self.domain_bias = None

        # Initialize domain-specific weights
        for i in range(n_domain):
            # e.g., xavier init for A, zeros for B or any scheme you prefer
            kernel_initializer(self.a_kernel[i])
            bias_initializer(self.b_kernel[i])  # we can re-use the bias init or use a specialized one
            if self.domain_bias is not None:
                bias_initializer(self.domain_bias[i])

        # Dropout for the main MLP output
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, domain_indicator):
        """
        Args:
            x: [batch_size, in_dim] input features
            domain_indicator: [batch_size] domain index
                              we assume the entire batch is from the same domain
        """
        # We'll pick the domain idx from the first sample
        # (assuming homogeneous domain in the batch)
        domain_idx = int(domain_indicator[0].item())

        # ------ 1) Main MLP path ------
        # outputs = x * kernel + bias
        outputs = x @ self.kernel
        if self.use_bias:
            outputs = outputs + self.bias  # shape [B, units]
        # optional dropout
        outputs = self.dropout(outputs)

        # ------ 2) Domain-specific LoRA path ------
        # domain_a_kernel: shape [in_dim, lora_r]
        domain_a_kernel = self.a_kernel[domain_idx]  # [in_dim, lora_r]
        # domain_b_kernel: shape [lora_r, units]
        domain_b_kernel = self.b_kernel[domain_idx]  # [lora_r, units]

        # domain_outputs = x * A * B
        # shape after x @ A => [B, lora_r]
        domain_outputs = x @ domain_a_kernel
        # shape after domain_outputs @ B => [B, units]
        domain_outputs = domain_outputs @ domain_b_kernel

        # domain bias if any
        if self.domain_bias is not None:
            domain_outputs = domain_outputs + self.domain_bias[domain_idx]

        # combine domain path with main
        outputs = outputs + domain_outputs

        # ------ 3) Activation ------
        if self.activation_fn is not None:
            outputs = self.activation_fn(outputs)

        return outputs

    def _get_activation(self, activation):
        """Map a string or callable to a PyTorch activation function."""
        if activation is None:
            return None
        if isinstance(activation, str):
            if activation.lower() == 'relu':
                return F.relu
            elif activation.lower() == 'sigmoid':
                return torch.sigmoid
            elif activation.lower() == 'tanh':
                return torch.tanh
            # add more if needed
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        elif callable(activation):
            return activation
        else:
            raise ValueError(f"Invalid activation: {activation}")
