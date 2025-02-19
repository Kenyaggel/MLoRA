import torch
import torch.nn as nn
import torch.nn.functional as F

class MLoRAFCN(nn.Module):
    """
    A PyTorch replication of your TF-based MLoRAFCN layer that:
      1) Applies a main linear transformation with trainable kernel+bias
      2) Adds domain-specific LoRA transformations (A, B, bias) which
         are *frozen* (trainable=False in TF).
      3) Optionally applies multiple LoRA sets if `lora_weight_list` is non-empty.
      4) Uses an activation function at the end if specified.
      5) Uses dropout on the main MLP path.

    Args:
        n_domain (int): number of domains
        units (int): output dimension of this FC layer
        n_domain_1 (int): placeholder (unused in example)
        n_domain_2 (int): placeholder (unused in example)
        in_dim (int): input dimension (must be known)
        activation (str or callable): activation function
        use_bias (bool): whether to use bias in the main MLP path
        kernel_initializer (callable): function to initialize main kernel
        bias_initializer (callable): function to initialize main/bias or domain weights
        lora_r (int): LoRA rank
        lora_reduce (int): if >=1 and lora_r <1, sets lora_r = max(units/lora_reduce,1)
        dropout_rate (float): dropout probability for main path
        is_finetune (bool): placeholder indicating fine-tuning mode
        lora_weight_list (list[float]): optional scaling factors for multiple LoRA sets
    """

    def __init__(self,
                 n_domain,
                 units,
                 n_domain_1=-1,
                 n_domain_2=-1,
                 in_dim=None,          # PyTorch needs an explicit input dimension
                 activation=None,
                 use_bias=True,
                 kernel_initializer=nn.init.xavier_uniform_,
                 bias_initializer=nn.init.zeros_,
                 kernel_regularizer=None,  # placeholders, not directly used in example
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 lora_r=4,
                 lora_reduce=-1,
                 dropout_rate=0.5,
                 is_finetune=False,
                 lora_weight_list=None):
        super(MLoRAFCN, self).__init__()

        # Basic configs
        self.n_domain = n_domain
        self.n_domain_1 = n_domain_1
        self.n_domain_2 = n_domain_2
        self.units = units
        self.in_dim = in_dim
        self.activation_fn = self._get_activation(activation)
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.is_finetune = is_finetune  # If needed, you can do partial param freezing
        if lora_r < 1 and lora_reduce >= 1:
            self.lora_r = max(int(units / lora_reduce), 1)
        else:
            self.lora_r = lora_r

        # We'll store an optional list of scaling factors for multiple LoRA sets
        self.lora_weight_list = lora_weight_list if lora_weight_list is not None else []

        # ====== Main trainable kernel & bias ====== #
        # kernel: shape [in_dim, units]
        self.kernel = nn.Parameter(torch.empty(in_dim, units))
        kernel_initializer(self.kernel)
        if self.use_bias:
            # main bias: shape [units]
            self.bias = nn.Parameter(torch.empty(units))
            bias_initializer(self.bias)
        else:
            self.bias = None

        # ====== Domain-specific LoRA (frozen) ====== #
        # A_Kernel: shape [n_domain, in_dim, lora_r]
        self.a_kernel = nn.Parameter(torch.empty(n_domain, in_dim, self.lora_r), requires_grad=False)
        # B_Kernel: shape [n_domain, lora_r, units]
        self.b_kernel = nn.Parameter(torch.empty(n_domain, self.lora_r, units), requires_grad=False)
        if self.use_bias:
            # domain_bias: shape [n_domain, units]
            self.domain_bias = nn.Parameter(torch.empty(n_domain, units), requires_grad=False)
        else:
            self.domain_bias = None

        # Initialize the domain-specific parameters (frozen)
        for d_idx in range(n_domain):
            bias_initializer(self.a_kernel[d_idx])
            bias_initializer(self.b_kernel[d_idx])
            if self.domain_bias is not None:
                bias_initializer(self.domain_bias[d_idx])

        # If we have multiple LoRA sets, replicate them
        # in TF code, you used `if not self.lora_weight_list` vs. else
        # We'll interpret that to mean we can hold multiple sets of A/B/bias in lists:
        self.a_kernel_list = []
        self.b_kernel_list = []
        self.domain_bias_list = []
        if len(self.lora_weight_list) > 0:
            # Possibly you store multiple sets of a/b/bias
            # The original TF code references e.g. self.a_kernel_list[i], etc.
            # We'll assume each set has shape [n_domain, in_dim, lora_r], [n_domain, lora_r, units], etc.
            # If that's not accurate, adapt accordingly.
            for i, lw in enumerate(self.lora_weight_list):
                a_ = nn.Parameter(torch.empty(n_domain, in_dim, self.lora_r), requires_grad=False)
                b_ = nn.Parameter(torch.empty(n_domain, self.lora_r, units), requires_grad=False)
                bias_ = None
                if self.use_bias:
                    bias_ = nn.Parameter(torch.empty(n_domain, units), requires_grad=False)
                # init them
                for d_idx in range(n_domain):
                    bias_initializer(a_[d_idx])
                    bias_initializer(b_[d_idx])
                    if bias_ is not None:
                        bias_initializer(bias_[d_idx])
                self.a_kernel_list.append(a_)
                self.b_kernel_list.append(b_)
                self.domain_bias_list.append(bias_)

        # Create dropout for the main path
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x, domain_indicator):
        """
        x: [batch_size, in_dim]
        domain_indicator: [batch_size], entire batch from same domain => pick domain_indicator[0].
        """
        # 1) Main MLP path
        # outputs = x * kernel + bias
        outputs = x @ self.kernel
        if self.use_bias:
            outputs = outputs + self.bias
        outputs = self.dropout(outputs)

        # 2) Domain-specific LoRA
        # We'll pick the domain index from the first sample
        domain_idx = int(domain_indicator[0].item())

        # If we have NO extra LoRA sets
        if len(self.lora_weight_list) == 0:
            # domain_a_kernel: shape [in_dim, lora_r]
            domain_a_kernel = self.a_kernel[domain_idx]       # [in_dim, lora_r]
            domain_b_kernel = self.b_kernel[domain_idx]       # [lora_r, units]
            domain_outputs = x @ domain_a_kernel
            domain_outputs = domain_outputs @ domain_b_kernel
            if self.use_bias:
                domain_outputs = domain_outputs + self.domain_bias[domain_idx]
            outputs += domain_outputs

        else:
            # Multiple LoRA sets
            for i, lw in enumerate(self.lora_weight_list):
                domain_a_kernel = self.a_kernel_list[i][domain_idx]  # [in_dim, lora_r]
                domain_b_kernel = self.b_kernel_list[i][domain_idx]  # [lora_r, units]
                domain_outputs = x @ domain_a_kernel
                domain_outputs = domain_outputs @ domain_b_kernel
                if self.use_bias:
                    domain_outputs = domain_outputs + self.domain_bias_list[i][domain_idx]
                outputs += domain_outputs * lw

        # 3) Activation
        if self.activation_fn is not None:
            outputs = self.activation_fn(outputs)

        return outputs

    def _get_activation(self, activation):
        """Utility to map string/callable to a PyTorch activation function."""
        if activation is None:
            return None
        if isinstance(activation, str):
            act_lower = activation.lower()
            if act_lower == 'relu':
                return F.relu
            elif act_lower == 'sigmoid':
                return torch.sigmoid
            elif act_lower == 'tanh':
                return torch.tanh
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        elif callable(activation):
            return activation
        else:
            raise ValueError(f"Invalid activation: {activation}")
