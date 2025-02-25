import tensorflow as tf
from tensorflow.python.keras import layers
from deepctr.layers.activation import activation_layer

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
                 lora_r = 4,
                 num_experts=2,  # << new: number of experts
                 use_gate=True,  # New field
                 expert_index=None,  # New field
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
        self.lora_r = lora_r
        self.num_experts = num_experts
        self.use_gate = use_gate
        self.expert_index = expert_index
        self.activation_layers = []

        # We'll store sub-layers for gating network and the final dense layers
        self.gate_dense = None
        self.experts_A = []
        self.experts_B = []
        self.W_base = []
        self.b_base = []  # base bias for each layer
        self.built_experts = False  # we will build them once input shape is known.
        # Domain-specific bias (one per layer), shape = (n_domain, hidden_dim)
        self.lora_bias = []



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

            # Base bias
            b = self.add_weight(
                name=f'b_base_{layer_idx}',
                shape=(hidden_dim,),
                initializer='zeros',
                trainable=True
            )
            self.b_base.append(b)
            print("lora_r", self.lora_r)
            print("lora_reduce", self.lora_reduce)
            if self.lora_r < 1 and self.lora_reduce >= 1:
                lora_r_layer = max(int(hidden_dim/self.lora_reduce), 1)
            else:
                lora_r_layer = self.lora_r

            # Experts for current layer
            A_experts = []
            B_experts = []
            lora_biases = []
            for i in range(self.num_experts):
                A = self.add_weight(
                    name=f'A_{layer_idx}_{i}',
                    shape=(prev_dim, lora_r_layer),
                    initializer='glorot_uniform',
                    trainable=True
                )
                B = self.add_weight(
                    name=f'B_{layer_idx}_{i}',
                    shape=(lora_r_layer, hidden_dim),
                    initializer='zeros',
                    trainable=True
                )
                # Domain-specific bias for each layer: shape = (n_domain, hidden_dim)
                lora_b = self.add_weight(
                    name=f'b_lora_{layer_idx}_{i}',
                    shape=(hidden_dim),
                    initializer='zeros',
                    trainable=True
                )
                A_experts.append(A)
                B_experts.append(B)
                lora_biases.append(lora_b)

            self.experts_A.append(A_experts)
            self.experts_B.append(B_experts)
            self.lora_bias.append(lora_biases)

            prev_dim = hidden_dim  # Update for next layer

        self.dropout_layers = [tf.keras.layers.Dropout(self.dnn_dropout, seed=self.seed + i) for i in
                               range(len(self.dnn_hidden_units))]
        self.activation_layers = [activation_layer(self.activation) for _ in range(len(self.dnn_hidden_units))]
        # Define the gating network
        self.gate_dense = tf.keras.layers.Dense(
            self.num_experts, activation='softmax', name='moe_gate'
        )

        self.built_experts = True
        super(MloraMoE, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        """
        inputs: [dnn_input, domain_input_layer]
          - dnn_input: (batch_size, input_dim)
          - domain_input_layer: (batch_size,) or (batch_size, embedding_dim)
        """
        dnn_input, domain_input = inputs

        # If domain_input is IDs, embed them (else skip if already an embedding)
        domain_emb = tf.keras.layers.Embedding(
            input_dim=self.n_domain, output_dim=4
        )(domain_input)  # (batch_size, 4)
        domain_emb = tf.keras.layers.Flatten()(domain_emb)  # (batch_size, 4)

        # If using gating, compute gate scores once (or you can do layer-wise if you prefer)
        gate_scores = None
        if self.use_gate:
            # For demonstration, gate on domain_emb only
            gate_scores = self.gate_dense(domain_emb)  # (batch_size, num_experts)
            gate_scores = tf.expand_dims(gate_scores, axis=-1)  # (batch_size, num_experts, 1)

        # print("lora bias", self.lora_bias)
        # print("experts_A", self.experts_A)
        # print("experts_B", self.experts_B)
        # print("W_base", self.W_base)
        # print("b_base", self.b_base)
        # print("domain_input", domain_input)
        # Forward pass through hidden layers
        x = dnn_input
        for layer_idx, hidden_dim in enumerate(self.dnn_hidden_units):
            # Backbone output
            base_out = tf.matmul(x, self.W_base[layer_idx])  # (batch_size, hidden_dim)
            base_out = tf.nn.bias_add(base_out, self.b_base[layer_idx])
            base_out = self.dropout_layers[layer_idx](base_out,training=training)

            # Compute each expert
            expert_outputs = []
            for i in range(self.num_experts):
                out_i = tf.matmul(x, self.experts_A[layer_idx][i])  # (batch_size, lora_reduce)
                out_i = tf.matmul(out_i, self.experts_B[layer_idx][i])  # (batch_size, hidden_dim)
                # domain_bias = tf.nn.embedding_lookup(self.lora_bias[layer_idx][i], domain_input)
                # out_i = out_i + domain_bias  # shape (batch_size, hidden_dim)
                # print("domain_input", domain_input)
                # print("domain_emb", domain_emb)
                # print("lora_bias", self.lora_bias[layer_idx][i])
                # print("A", self.experts_A[layer_idx][i].shape)
                # print("B", self.experts_B[layer_idx][i].shape)
                # if layer_idx < len(self.dnn_hidden_units) - 1:
                out_i = tf.nn.bias_add(out_i, self.lora_bias[layer_idx][i])
                # else:
                #
                #     out_i = tf.nn.bias_add(out_i, self.lora_bias[layer_idx+1][i])  # shape (batch_size, hidden_dim)
                expert_outputs.append(out_i)

            if self.use_gate:
                # Mixture of experts
                expert_stack = tf.stack(expert_outputs, axis=1)  # (batch_size, num_experts, hidden_dim)
                # Expand gate scores to match hidden_dim
                gate_scores_expanded = tf.tile(gate_scores, [1, 1, hidden_dim])
                moe_out = tf.reduce_sum(expert_stack * gate_scores_expanded, axis=1)
                x = base_out + moe_out
            else:
                # No gating
                if self.expert_index is not None:
                    # Use backbone + the single expert
                    x = base_out + expert_outputs[self.expert_index]
                else:
                    # Only backbone
                    x = base_out

            # # Activation & Dropout
            # x = tf.keras.layers.Activation(self.activation)(x)
            try:
                x = self.activation_layers[layer_idx](x, training=training)
            except TypeError as e:  # TypeError: call() got an unexpected keyword argument 'training'
                print("make sure the activation function use training flag properly", e)
                x = self.activation_layers[layer_idx](x)


            # if self.dnn_dropout > 0:
            #     x = tf.keras.layers.Dropout(self.dnn_dropout)(x)


        return x

    ############################################################################
    # Below are the methods to freeze/unfreeze different parts of this layer.  #
    ############################################################################

    def freeze_backbone(self, freeze=True):
        """
        Freeze or unfreeze the backbone weights (W_base).
        freeze=True makes them non-trainable; freeze=False makes them trainable.
        """
        for w in self.W_base:
            w._trainable = not freeze  # or w.trainable = not freeze (if Weight objects allow)

    def freeze_gating(self, freeze=True):
        """
        Freeze or unfreeze the gating network (gate_dense).
        """
        self.gate_dense.trainable = not freeze

    def freeze_experts(self, freeze=True):
        """
        Freeze or unfreeze ALL experts (A_i and B_i for each expert i, each layer).
        """
        for layer_idx in range(len(self.experts_A)):
            for i in range(self.num_experts):
                self.experts_A[layer_idx][i]._trainable = not freeze
                self.experts_B[layer_idx][i]._trainable = not freeze

    def freeze_expert(self, expert_index, freeze=True):
        """
        Freeze or unfreeze a specific expert across all layers.
        """
        for layer_idx in range(len(self.experts_A)):
            self.experts_A[layer_idx][expert_index]._trainable = not freeze
            self.experts_B[layer_idx][expert_index]._trainable = not freeze
