import torch
import torch.nn as nn
import torch.nn.functional as F

class MLoRAFCN(nn.Module):
    """
    A PyTorch version of the TF-based MLoRAFCN layer that:
      - Has a frozen main kernel + bias.
      - Has trainable domain-specific LoRA (A, B, bias).
      - Can also include multiple additional LoRA sets
        (with different ranks) & weights for each set.
      - Applies dropout on the main path.
      - Merges domain outputs with the main outputs and
        applies an activation at the end if specified.

    Args:
        n_domain (int): Number of domains.
        units (int): Output dimension of the layer.
        n_domain_1 (int): unused placeholder
        n_domain_2 (int): unused placeholder
        in_dim (int): Must be specified for PyTorch (dimension of input features).
        activation (str or callable): Activation function (e.g. "relu").
        use_bias (bool): Whether to use a bias in the main linear path.
        kernel_initializer (callable): Used for main kernel initialization.
        bias_initializer (callable): Used for main bias or LoRA A/B init.
        lora_r (int): The base rank for LoRA.
        lora_reduce (int): If >=1 and lora_r <1, override lora_r.
        lora_reduce_list (list[int]): Additional "alpha" values for multiple LoRA sets.
        lora_weight_list (list[float]): Additional scaling factors for each extra LoRA set.
        dropout_rate (float): Probability of dropout in main path.
        is_finetune (bool): If True, might freeze some parameters or do special logic.
    """

    def __init__(self,
                 n_domain,
                 units,
                 n_domain_1=-1,
                 n_domain_2=-1,
                 in_dim=None,  # Must be provided for PyTorch
                 activation=None,
                 use_bias=True,
                 kernel_initializer=nn.init.xavier_uniform_,
                 bias_initializer=nn.init.zeros_,
                 kernel_regularizer=None,  # placeholders
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 lora_r=4,
                 lora_reduce=-1,
                 lora_reduce_list=None,
                 lora_weight_list=None,
                 dropout_rate=0.5,
                 is_finetune=False):
        super(MLoRAFCN, self).__init__()

        self.n_domain = n_domain
        self.n_domain_1 = n_domain_1
        self.n_domain_2 = n_domain_2
        self.units = units
        self.in_dim = in_dim
        self.activation_fn = self._get_activation(activation)
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.is_finetune = is_finetune

        # Possibly override lora_r if lora_reduce is given
        if lora_r < 1 and lora_reduce >= 1:
            self.lora_r = max(int(units / lora_reduce), 1)
        else:
            self.lora_r = lora_r

        # Build lora_r_list from lora_reduce_list (if any)
        if lora_reduce_list is None:
            lora_reduce_list = []
        self.lora_r_list = []
        for alpha in lora_reduce_list:
            # same logic as above
            r_val = self.lora_r
            if self.lora_r < 1 and alpha >= 1:
                r_val = max(int(units / alpha), 1)
            else:
                # or you might interpret each alpha differently
                r_val = max(int(units / alpha), 1)
            self.lora_r_list.append(r_val)

        # Keep a float scaling factor for each extra LoRA set
        if lora_weight_list is None:
            lora_weight_list = []
        self.lora_weight_list = [torch.tensor(w, dtype=torch.float32) for w in lora_weight_list]

        # --------------- Main (frozen) kernel & bias --------------- #
        self.kernel = nn.Parameter(torch.empty(in_dim, units), requires_grad=False)
        kernel_initializer(self.kernel)
        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(units), requires_grad=False)
            bias_initializer(self.bias)
        else:
            self.bias = None

        # --------------- Primary LoRA domain parameters --------------- #
        # A_Kernel: shape [n_domain, in_dim, self.lora_r] (trainable)
        self.a_kernel = nn.Parameter(torch.empty(n_domain, in_dim, self.lora_r), requires_grad=True)
        kernel_initializer(self.a_kernel)
        # B_Kernel: shape [n_domain, self.lora_r, units] (trainable)
        self.b_kernel = nn.Parameter(torch.empty(n_domain, self.lora_r, units), requires_grad=True)
        bias_initializer(self.b_kernel)

        if self.use_bias:
            # domain_bias: shape [n_domain, units]
            self.domain_bias = nn.Parameter(torch.empty(n_domain, units), requires_grad=True)
            bias_initializer(self.domain_bias)
        else:
            self.domain_bias = None

        # --------------- Additional LoRA sets --------------- #
        self.a_kernel_list = nn.ParameterList()
        self.b_kernel_list = nn.ParameterList()
        self.domain_bias_list = nn.ParameterList()  # only if use_bias
        for i, r in enumerate(self.lora_r_list):
            # For each extra set, create A/B plus domain bias if needed
            a_ = nn.Parameter(torch.empty(n_domain, in_dim, r), requires_grad=True)
            b_ = nn.Parameter(torch.empty(n_domain, r, units), requires_grad=True)
            kernel_initializer(a_)
            bias_initializer(b_)

            self.a_kernel_list.append(a_)
            self.b_kernel_list.append(b_)

            if self.use_bias:
                db_ = nn.Parameter(torch.empty(n_domain, units), requires_grad=True)
                bias_initializer(db_)
                self.domain_bias_list.append(db_)
            else:
                # If no bias, append None or skip
                self.domain_bias_list.append(nn.Parameter(torch.empty(0), requires_grad=False))

        # Create dropout for the main path
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x, domain_indicator):
        """
        x: [batch_size, in_dim]
        domain_indicator: [batch_size], assume same domain for entire batch => domain_indicator[0].
        """
        # ---- 1) Main (frozen) path ----
        # outputs = x @ kernel + bias
        outputs = x @ self.kernel
        if self.use_bias:
            outputs = outputs + self.bias
        outputs = self.dropout(outputs)

        # ---- 2) Domain index ----
        domain_idx = int(domain_indicator[0].item())

        # ---- 3) Primary LoRA A/B path ----
        # domain_a_kernel: [in_dim, lora_r]
        domain_a_kernel = self.a_kernel[domain_idx]
        # domain_b_kernel: [lora_r, units]
        domain_b_kernel = self.b_kernel[domain_idx]

        domain_outputs = x @ domain_a_kernel
        domain_outputs = domain_outputs @ domain_b_kernel

        if self.use_bias:
            domain_outputs = domain_outputs + self.domain_bias[domain_idx]

        outputs = outputs + domain_outputs

        # ---- 4) Additional LoRA sets ----
        # If we have self.lora_weight_list, each set has a_ker, b_ker, (maybe domain_bias)
        for i, weight_factor in enumerate(self.lora_weight_list):
            a_ker = self.a_kernel_list[i][domain_idx]  # shape [in_dim, r]
            b_ker = self.b_kernel_list[i][domain_idx]  # shape [r, units]

            extra_outputs = x @ a_ker
            extra_outputs = extra_outputs @ b_ker
            if self.use_bias:
                extra_outputs = extra_outputs + self.domain_bias_list[i][domain_idx]

            outputs = outputs + extra_outputs * weight_factor

        # ---- 5) Activation ----
        if self.activation_fn is not None:
            outputs = self.activation_fn(outputs)

        return outputs

    def _get_activation(self, activation):
        """Utility to map a string/callable activation to PyTorch function."""
        if activation is None:
            return None
        if isinstance(activation, str):
            a = activation.lower()
            if a == 'relu':
                return F.relu
            elif a == 'sigmoid':
                return torch.sigmoid
            elif a == 'tanh':
                return torch.tanh
            else:
                raise ValueError(f"Unknown activation: {activation}")
        elif callable(activation):
            return activation
        else:
            raise ValueError(f"Invalid activation: {activation}")
