import argparse
import json

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python import train
from model_zoo.DeepCTR import DeepCTR
from model_zoo.DeepCTR import DeepCTR_LORA
from model_zoo.Star import Star

from model_zoo.MLoRA import MLoRA
from model_zoo.lora_moe import MloraMoE
from utils import MultiDomainDataset
from tensorflow.python.keras import layers, backend, callbacks
from utils.auc import AUC

def in_name_list(x, name_list):
    for n in name_list:
        if n in x:
            return True
    return False

def layer_matching(prelayer_name,layer_name):
    if prelayer_name == layer_name:
        return True
    elif prelayer_name == "dense" and layer_name == "dense_2":
        return True
    elif prelayer_name == "dense" and layer_name == "dense_1":
        return True
    elif prelayer_name == "partitioned_norm" and layer_name == "partitioned_norm_1":
        return True
    elif prelayer_name == "m_lo_rafcn" and layer_name == "m_lo_rafcn_3":
        return True
    elif prelayer_name == "m_lo_rafcn_1" and layer_name == "m_lo_rafcn_4":
        return True
    elif prelayer_name == "m_lo_rafcn_2" and layer_name == "m_lo_rafcn_5":
        return True
    elif prelayer_name == "star_fcn" and layer_name == "star_fcn_3":
        return True
    elif prelayer_name == "star_fcn_1" and layer_name == "star_fcn_4":
        return True
    elif prelayer_name == "star_fcn_2" and layer_name == "star_fcn_5":
        return True
    else:
        return False

def freeze_mlora_parts(model, freeze_dict):
    """
    Given a dictionary like:
        freeze_dict = {
            "backbone": True/False,
            "gate": True/False,
            "all_experts": True/False,
            "expert_index": int or None  # for training a single expert
        }
    This function finds all MloraMoE layers in the model and applies the freeze/unfreeze calls.
    """
    for layer in model.model.layers:
        # Check by type or by name
        if isinstance(layer, MloraMoE):  # or if "MloraMoE" in layer.name:
            # 1) Freeze or unfreeze the backbone
            if "backbone" in freeze_dict:
                layer.freeze_backbone(freeze=freeze_dict["backbone"])
            # 2) Freeze or unfreeze the gating network
            if "gate" in freeze_dict:
                layer.freeze_gating(freeze=freeze_dict["gate"])
            # 3) Freeze or unfreeze all experts
            if "all_experts" in freeze_dict:
                layer.freeze_experts(freeze=freeze_dict["all_experts"])
            # 4) If we want to unfreeze only ONE specific expert, freeze the rest
            if "expert_index" in freeze_dict and freeze_dict["expert_index"] is not None:
                num_experts = layer.num_experts
                # Freeze all experts first
                layer.freeze_experts(True)
                # Unfreeze only the selected one
                idx = freeze_dict["expert_index"]
                if 0 <= idx < num_experts:
                    layer.freeze_expert(expert_index=idx, freeze=False)

def main(config):
    tf.set_random_seed(config['dataset']['seed'])
    c = tf.ConfigProto()
    c.gpu_options.allow_growth = True
    sess = tf.Session(config=c)
    K.set_session(sess)

    # Load Dataset
    dataset = MultiDomainDataset(config['dataset'])

    # Choose Model
    model = None
    deep_ctr_lora_list = ['mlp_lora', 'wdl_lora', 'nfm_lora', 'autoint_lora','deepfm_lora','dcn_lora','xdeepfm_lora','fibinet_lora','pnn_lora']
    deep_ctr_list = ['mlp_single', 'wdl_single', 'nfm_single', 'autoint_single','deepfm_single','dcn_single','xdeepfm_single','fibinet_single','pnn_single']
    if 'mlora' in config['model']['name']:
        model = MLoRA(dataset, config)
    elif 'star' in config['model']['name']:
        model = Star(dataset, config)
    # deep learning methods
    elif in_name_list(config['model']['name'], deep_ctr_list):
        model = DeepCTR(dataset, config)
    # deep learning methods with MLoRA
    elif in_name_list(config['model']['name'], deep_ctr_lora_list):
        model = DeepCTR_LORA(dataset, config)
    else:
        print("model: {} not found".format(config['model']['name']))

    # get model flops
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    options = tf.profiler.ProfileOptionBuilder.float_operation()
    options['output'] = 'none'  # Redirect output to nowhere
    flops = tf.profiler.profile(sess.graph,options=options)
    print("FLOPS:", flops.total_float_ops)
    sess.close()

    ###########################################################################
    # 1) Train ONLY the backbone
    ###########################################################################
    # Freeze gating + all experts, unfreeze backbone
    freeze_dict = {
        "backbone": False,  # train backbone
        "gate": True,  # freeze gate
        "all_experts": True,  # freeze experts
        "expert_index": None
    }
    freeze_mlora_parts(model, freeze_dict)

    # Now tell each MloraMoE layer to skip gating and skip experts in forward pass
    for layer in model.model.layers:
        if isinstance(layer, MloraMoE):
            layer.use_gate = False
            layer.expert_index = None  # means only backbone
    if model.train_config['optimizer'] == 'adam':
        opt = train.AdamOptimizer(learning_rate=model.train_config['learning_rate'])
    else:
        opt = model.train_config['optimizer']
    # K.clear_session()
    # tf.reset_default_graph()
    model.model.compile(loss=model.train_config['loss'], optimizer=opt, metrics=[AUC(num_thresholds=500, name="AUC")])

    print('#' * 25, "Training ONLY backbone", '#' * 25)
    model.train()

    print("Test Result after training backbone: ")
    avg_loss, avg_auc, domain_loss, domain_auc = model.val_and_test("test")

    ###########################################################################
    # 2) Train each expert (one by one)
    ###########################################################################
    num_experts = config['model']['num_experts']
    print('#' * 25, "Training experts individually", '#' * 25)
    for i in range(num_experts):
        print("Training expert", i)
        # Freeze backbone + gate, unfreeze only expert i
        freeze_dict = {
            "backbone": True,
            "gate": True,
            "all_experts": None,  # not used
            "expert_index": i  # unfreeze only expert i
        }
        freeze_mlora_parts(model, freeze_dict)

        # Forward pass uses only that single expert (no gating, no backbone)
        for layer in model.model.layers:
            if isinstance(layer, MloraMoE):
                layer.use_gate = False
                layer.expert_index = i

        domain_id = i % model.n_domain
        model.reset_early_stop()  # reset best metric for each expert
        # K.clear_session()
        model.model.compile(loss=model.train_config['loss'], optimizer=opt, metrics=[AUC(num_thresholds=500, name="AUC")])
        model.train(domain_ids=[domain_id])  # trains only expert i

    # Evaluate after experts training
    print("Test Result after training experts: ")
    avg_loss, avg_auc, domain_loss, domain_auc = model.val_and_test("test")

    ###########################################################################
    # 3) Train ONLY the gate
    ###########################################################################
    freeze_dict = {
        "backbone": True,  # freeze
        "gate": False,  # unfreeze
        "all_experts": True,  # freeze all experts
        "expert_index": None
    }
    freeze_mlora_parts(model, freeze_dict)

    # Forward pass uses gating mixture, but experts are frozen (so gate is the only trainable part).
    for layer in model.model.layers:
        if isinstance(layer, MloraMoE):
            layer.use_gate = True
            layer.expert_index = None  # gating with all experts

    print('#' * 25, "Training ONLY gate", '#' * 25)
    model.reset_early_stop()  # reset best metric for gate training
    # K.clear_session()
    model.model.compile(loss=model.train_config['loss'], optimizer=opt, metrics=[AUC(num_thresholds=500, name="AUC")])
    model.train()

    # Final evaluation
    print("Test Result after training gate: ")
    avg_loss, avg_auc, domain_loss, domain_auc = model.val_and_test("test")

    # Save final results
    model.save_result(avg_loss, avg_auc, domain_loss, domain_auc)
    w_auc = model._weighted_auc("test", domain_auc)
    return avg_loss, avg_auc, domain_loss, domain_auc, w_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Train config file", required=False, default="config/Movielens/autoint_mlora.json")
    args = parser.parse_args()
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    main(config)
