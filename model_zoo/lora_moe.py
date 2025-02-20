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
        self.W_base = None
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
        input_shape: (list) shape of [dnn_input, domain_input_layer],
                     or something similar. We'll assume:
                     dnn_input: (batch_size, input_dim)
                     domain_input_layer: (batch_size,) or (batch_size, embedding_dim)
        """
        # For consistency, let's do:
        dnn_input_shape = input_shape[0]
        input_dim = int(dnn_input_shape[-1])

        # Create base weight W_base
        self.W_base = self.add_weight(
            name='W_base',
            shape=(input_dim, self.dnn_hidden_units[0]),
            initializer='glorot_uniform',
            trainable=True
        )

        # Create LoRA experts: A_i, B_i
        for i in range(self.num_experts):
            A = self.add_weight(
                name=f'A_{i}',
                shape=(input_dim, self.lora_reduce),
                initializer='glorot_uniform',
                trainable=True
            )
            B = self.add_weight(
                name=f'B_{i}',
                shape=(self.lora_reduce, self.dnn_hidden_units[0]),
                initializer='zeros',
                trainable=True
            )
            self.experts_A.append(A)
            self.experts_B.append(B)

        # Simple gating: a Dense => softmax
        # If domain_input_layer is an embedding, we can pass it to gating
        # along with the dnn_input, or just domain.
        # For now, let’s do gating on domain alone:
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

        # Base output = dnn_input * W_base
        base_out = tf.matmul(dnn_input, self.W_base)  # (batch_size, hidden_units[0])

        # Experts output
        expert_outputs = []
        for i in range(self.num_experts):
            out_i = tf.matmul(dnn_input, self.experts_A[i])  # (batch_size, lora_reduce)
            out_i = tf.matmul(out_i, self.experts_B[i])  # (batch_size, hidden_units[0])
            expert_outputs.append(out_i)

        expert_outputs = tf.stack(expert_outputs, axis=1)  # => (batch_size, num_experts, hidden_units[0])

        # Weighted sum across experts
        gate_scores = tf.tile(gate_scores, [1, 1, self.dnn_hidden_units[0]])  # (batch_size, num_experts, hidden[0])
        moe_out = tf.reduce_sum(expert_outputs * gate_scores, axis=1)  # (batch_size, hidden_units[0])

        # Combine base_out + MoE
        combined_out = base_out + moe_out  # (batch_size, hidden_units[0])

        # Pass through post_layers (the rest of the DNN)
        dnn_out = combined_out
        for layer in self.post_layers:
            dnn_out = layer(dnn_out)

        return dnn_out
