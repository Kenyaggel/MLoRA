import tensorflow as tf
from tensorflow.python.keras import layers


class MloraMoE(layers.Layer):
    """
    A Mixture-of-Experts LoRA layer that:

    - Takes a main DNN input plus a domain input (like Mlora).
    - Has multiple low-rank adapter 'experts' (A_i, B_i).
    - Uses a gating network that outputs mixture weights for each expert.
    """

    def __init__(self,
                 n_domain,
                 dnn_hidden_units,
                 activation='relu',
                 l2_reg_dnn=0.,
                 dnn_dropout=0.,
                 use_bn=False,
                 seed=1024,
                 lora_reduce=4,
                 num_experts=2,  # << new: number of experts
                 **kwargs):
        super(MloraMoE, self).__init__(**kwargs)

        self.n_domain = n_domain
        self.dnn_hidden_units = dnn_hidden_units
        self.activation = activation
        self.l2_reg_dnn = l2_reg_dnn
        self.dnn_dropout = dnn_dropout
        self.use_bn = use_bn
        self.seed = seed
        self.lora_reduce = lora_reduce
        self.num_experts = num_experts

        # We'll store sub-layers for gating network and the final dense layers
        self.gate_dense = None
        self.experts_A = []
        self.experts_B = []
        self.W_base = []
        self.built_experts = False  # we will build them once input shape is known.

        # If you also want hidden layers after the MoE, you can define them here:
        self.post_layers = []
        for i, unit in enumerate(dnn_hidden_units):
            self.post_layers.append(
                tf.keras.layers.Dense(
                    unit,
                    activation=self.activation,
                    kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg_dnn),
                    name=f"post_dense_{i}"
                )
            )
            if dnn_dropout > 0:
                self.post_layers.append(tf.keras.layers.Dropout(self.dnn_dropout))

    def build(self, input_shape):
        """
        input_shape: (list) shape of [dnn_input, domain_input_layer]
        - dnn_input: (batch_size, input_dim)
        - domain_input_layer: (batch_size,) or (batch_size, embedding_dim)
        """
        dnn_input_shape = input_shape[0]
        input_dim = int(dnn_input_shape[-1])

        # Create base weight and LoRA experts for each layer in dnn_hidden_units
        prev_dim = input_dim
        for layer_idx, hidden_dim in enumerate(self.dnn_hidden_units):
            # Base weight for the current layer
            W = self.add_weight(
                name=f'W_base_{layer_idx}',
                shape=(prev_dim, hidden_dim),
                initializer='glorot_uniform',
                trainable=True
            )
            self.W_base.append(W)

            # Experts for current layer
            A_experts = []
            B_experts = []
            for i in range(self.num_experts):
                A = self.add_weight(
                    name=f'A_{layer_idx}_{i}',
                    shape=(prev_dim, self.lora_reduce),
                    initializer='glorot_uniform',
                    trainable=True
                )
                B = self.add_weight(
                    name=f'B_{layer_idx}_{i}',
                    shape=(self.lora_reduce, hidden_dim),
                    initializer='zeros',
                    trainable=True
                )
                A_experts.append(A)
                B_experts.append(B)

            self.experts_A.append(A_experts)
            self.experts_B.append(B_experts)

            prev_dim = hidden_dim  # Update for next layer

        # Define the gating network
        self.gate_dense = tf.keras.layers.Dense(
            self.num_experts, activation='softmax', name='moe_gate'
        )

        self.built_experts = True
        super(MloraMoE, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """
        inputs: [dnn_input, domain_input_layer]
        - dnn_input: (batch_size, input_dim)
        - domain_input_layer: (batch_size,) or an embedding
        """
        dnn_input, domain_input = inputs
        # Suppose domain_input is (batch_size, ) domain IDs.
        # If we want gating per-domain, we can embed them or convert them to one-hot.
        # For demonstration, let's embed domain_input with a small Dense:
        # If domain_input is already an embedding, skip this step.
        # Embed domain ID to a small vector
        domain_emb = tf.keras.layers.Embedding(
            input_dim=self.n_domain, output_dim=4)(domain_input)  # (batch_size, 4)
        domain_emb = tf.keras.layers.Flatten()(domain_emb)  # (batch_size, 4)

        # Gating:
        # Option 1) gating on domain_emb only
        # Option 2) gating on [dnn_input, domain_emb]
        # We'll pick domain_emb only, for clarity.
        gate_scores = self.gate_dense(domain_emb)  # (batch_size, num_experts)
        gate_scores = tf.expand_dims(gate_scores, axis=-1)  # => (batch_size, num_experts, 1)

        # Pass through all hidden layers dynamically
        x = dnn_input
        for layer_idx, hidden_dim in enumerate(self.dnn_hidden_units):
            # Compute base output
            base_out = tf.matmul(x, self.W_base[layer_idx])  # (batch_size, hidden_dim)

            # Compute expert outputs
            expert_outputs = []
            for i in range(self.num_experts):
                out_i = tf.matmul(x, self.experts_A[layer_idx][i])  # (batch_size, lora_reduce)
                out_i = tf.matmul(out_i, self.experts_B[layer_idx][i])  # (batch_size, hidden_dim)
                expert_outputs.append(out_i)

            # Stack expert outputs and compute weighted sum
            expert_outputs = tf.stack(expert_outputs, axis=1)  # (batch_size, num_experts, hidden_dim)
            gate_scores_expanded = tf.tile(gate_scores, [1, 1, hidden_dim])  # (batch_size, num_experts, hidden_dim)
            moe_out = tf.reduce_sum(expert_outputs * gate_scores_expanded, axis=1)  # (batch_size, hidden_dim)

            # Combine base output with MoE output
            x = base_out + moe_out

            # Apply activation and dropout
            x = tf.keras.layers.Activation(self.activation)(x)
            if self.dnn_dropout > 0:
                x = tf.keras.layers.Dropout(self.dnn_dropout)(x)

        return x
