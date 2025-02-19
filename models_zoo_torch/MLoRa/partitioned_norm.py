import torch
import torch.nn as nn
import torch.nn.functional as F

class PartitionedNorm(nn.Module):
    """
    A PyTorch replication of your TensorFlow-based PartitionedNorm logic.
    Each domain has its own gamma, beta, running_mean, and running_var,
    plus there is a shared gamma/beta that is combined with domain-specific
    parameters:
        final_gamma = shared_gamma * domain_gamma
        final_beta  = shared_beta  + domain_beta

    NOTE:
    - We assume each batch contains samples from exactly ONE domain,
      indicated by domain_indicator[0].
    - Running stats are updated using a momentum approach, just like BatchNorm.
    - If you want to freeze domain-specific stats, you can disable
      parameter gradients or updates accordingly.
    """

    def __init__(self,
                 n_domain,
                 dim=None,
                 momentum=0.99,
                 eps=1e-3,
                 trainable=True):
        """
        :param n_domain: Number of domains
        :param dim: Feature dimension along which to normalize
                    (must match the last dimension of x).
        :param momentum: Momentum factor for updating running stats
        :param eps: Small constant for numerical stability
        :param trainable: If False, parameters won't be learnable
                          (gamma/beta remain fixed).
        """
        super(PartitionedNorm, self).__init__()
        self.n_domain = n_domain
        self.dim = dim  # you must set or infer from your model
        self.momentum = momentum
        self.eps = eps

        # Domain-specific gamma/beta
        # Shape: [n_domain, dim]
        self.domain_gamma = nn.Parameter(torch.ones(n_domain, dim))
        self.domain_beta = nn.Parameter(torch.zeros(n_domain, dim))

        # Shared gamma/beta
        # Shape: [dim]
        self.shared_gamma = nn.Parameter(torch.ones(dim))
        self.shared_beta = nn.Parameter(torch.zeros(dim))

        # Running stats: shape [n_domain, dim]
        self.register_buffer('running_mean', torch.zeros(n_domain, dim))
        self.register_buffer('running_var', torch.ones(n_domain, dim))

        if not trainable:
            # Freeze parameters
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x, domain_indicator):
        """
        :param x: (B, dim) tensor to be normalized (batch along dim=0).
        :param domain_indicator: (B,) or (B,1) with the domain index
               for the batch. If each batch only has one domain,
               we use domain_indicator[0] to pick that domain’s stats.

        NOTE: This version assumes the entire batch is from one domain.
        If your batch can contain multiple domains, you'd need a more
        advanced grouping approach.
        """
        # If your data loader ensures each batch is domain-homogeneous,
        # we can just pick domain_indicator[0].
        # e.g. domain_idx = domain_indicator[0,0] in TF, but here:
        domain_idx = int(domain_indicator[0].item())

        # Gather domain-specific parameters
        d_gamma = self.domain_gamma[domain_idx]  # shape [dim]
        d_beta = self.domain_beta[domain_idx]    # shape [dim]

        # Combine with shared gamma/beta
        gamma = self.shared_gamma * d_gamma      # shape [dim]
        beta = self.shared_beta + d_beta         # shape [dim]

        # Retrieve running stats
        running_mean = self.running_mean[domain_idx]
        running_var = self.running_var[domain_idx]

        # Compute batch statistics
        if self.training:
            # mean, var over batch dimension (dim=0)
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            # Update running stats (like PyTorch BN)
            with torch.no_grad():
                self.running_mean[domain_idx] = (
                    (1.0 - self.momentum) * running_mean +
                    self.momentum * batch_mean
                )
                self.running_var[domain_idx] = (
                    (1.0 - self.momentum) * running_var +
                    self.momentum * batch_var
                )

            mean = batch_mean
            var = batch_var
        else:
            # In eval mode, use stored running stats
            mean = running_mean
            var = running_var

        # Normalize
        # y = (x - mean) / sqrt(var + eps) * gamma + beta
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * gamma + beta
        return out
