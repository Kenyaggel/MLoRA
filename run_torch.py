import argparse
import json
import os
import random

import torch
import torch.nn as nn
import numpy as np
import sys
print(sys.path)
import os
print(os.listdir('.'))
print(os.listdir('./models_zoo_torch'))
print(os.listdir('./models_zoo_torch/deepctr_torch'))
print(os.listdir('./models_zoo_torch/deepctr_torch/models'))

from models_zoo_torch.deepctr_torch.models import WDL, DeepFM, DCN, xDeepFM, NFM, AutoInt, PNN, FiBiNET
# from models_zoo_torch.deepctr_torch.models import wdl, deepfm, dcn, xdeepfm, nfm, autoint, pnn, fibinet
from models_zoo_torch.deepctr_torch.inputs import SparseFeat, DenseFeat#, get_feature_names

from models_zoo_torch.MLoRa.mlora import MLoRA
from utils_torch.dataset import MultiDomainDataset


# --------------------------------------------------------------------------
# Placeholders for your custom modules.
# You will need to adapt MLoRA, Star, MultiDomainDataset, etc. to PyTorch.
# --------------------------------------------------------------------------

# from model_zoo.Star_torch import Star  # example: your PyTorch version
# from model_zoo.MLoRA_torch import MLoRA  # example: your PyTorch version
# from utils_torch import MultiDomainDataset  # example: your PyTorch version


# --------------------------------------------------------------------------
# Example “DeepCTR” style wrappers for single-domain or multi-domain tasks.
# You may replace or extend to handle multiple domains in your dataset.
# --------------------------------------------------------------------------
class DeepCTR_Torch_Wrapper(nn.Module):
    """
    Example wrapper for a deepctr-torch model.
    This is just a minimal illustration; adapt to your multi-domain dataset.
    """

    def __init__(self, dataset, config):
        super(DeepCTR_Torch_Wrapper, self).__init__()
        self.config = config
        self.dataset = dataset

        # Prepare feature columns
        # Suppose your dataset returns a set of SparseFeat and DenseFeat features
        # self.sparse_features, self.dense_features = dataset.get_feature_columns()

        fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=10000, embedding_dim=8)
                                  for feat in self.sparse_features] + \
                                 [DenseFeat(feat, 1) for feat in self.dense_features]
        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns

        # Example: pick a deepctr-torch model by name
        # You might map from config['model']['name'] to a particular constructor
        # For demonstration, assume 'wdl_single' => WDL
        model_name = config['model']['name']
        if 'wdl_single' in model_name:
            self.model = WDL(linear_feature_columns, dnn_feature_columns,
                             task='binary', dnn_hidden_units=(128, 64))
        elif 'deepfm_single' in model_name:
            self.model = DeepFM(linear_feature_columns, dnn_feature_columns,
                                task='binary', dnn_hidden_units=(128, 64))
        elif 'nfm_single' in model_name:
            self.model = NFM(linear_feature_columns, dnn_feature_columns,
                             task='binary', dnn_hidden_units=(128, 64))
        elif 'autoint_single' in model_name:
            self.model = AutoInt(linear_feature_columns, dnn_feature_columns,
                                 task='binary', dnn_hidden_units=(128, 64))
        elif 'dcn_single' in model_name:
            self.model = DCN(linear_feature_columns, dnn_feature_columns,
                             task='binary', dnn_hidden_units=(128, 64))
        elif 'xdeepfm_single' in model_name:
            self.model = xDeepFM(linear_feature_columns, dnn_feature_columns,
                                 task='binary', dnn_hidden_units=(128, 64))
        elif 'fibinet_single' in model_name:
            self.model = FiBiNET(linear_feature_columns, dnn_feature_columns,
                                 task='binary', dnn_hidden_units=(128, 64))
        elif 'pnn_single' in model_name:
            self.model = PNN(linear_feature_columns, dnn_feature_columns,
                             task='binary', dnn_hidden_units=(128, 64))
        else:
            # fallback
            self.model = WDL(linear_feature_columns, dnn_feature_columns,
                             task='binary')

    def forward(self, x):
        # deepctr-torch models forward pass
        return self.model(x)

    def compile(self, optimizer, loss):
        # Typically you set up your PyTorch optimizer, criterion, etc.
        self.optimizer = optimizer
        self.loss_fn = loss

    def train_model(self, train_loader, epochs=5):
        self.train()
        for epoch in range(epochs):
            for batch_x, batch_y in train_loader:
                y_pred = self.model(batch_x)
                loss = self.loss_fn(y_pred, batch_y.float())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def evaluate_model(self, data_loader):
        # Evaluate on validation or test
        self.eval()
        total_loss = 0
        # If evaluating AUC, you can accumulate predictions and labels, then compute
        y_preds, y_trues = [], []
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                preds = self.model(batch_x)
                loss = self.loss_fn(preds, batch_y.float())
                total_loss += loss.item()
                y_preds.append(preds.detach().cpu().numpy())
                y_trues.append(batch_y.detach().cpu().numpy())
        # Suppose we measure AUC with scikit-learn
        from sklearn.metrics import roc_auc_score
        y_preds = np.concatenate(y_preds)
        y_trues = np.concatenate(y_trues)
        auc = roc_auc_score(y_trues, y_preds)
        avg_loss = total_loss / len(data_loader)
        return avg_loss, auc


# --------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------
def in_name_list(x, name_list):
    for n in name_list:
        if n in x:
            return True
    return False


def layer_matching(prelayer_name, layer_name):
    """
    Example placeholder for your custom layer matching logic.
    If you need to partially load weights or do specialized loading,
    you'll adapt it to match parameter names in PyTorch, which typically
    are like "model.layer.weight", "model.layer.bias", etc.
    """
    if prelayer_name == layer_name:
        return True
    # Additional custom matching
    return False


def get_directory_size(directory):
    """Return the total size of all files in the directory."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(config):
    # --------------------------
    # 1) Set random seeds
    # --------------------------
    seed = config['dataset']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --------------------------
    # 2) Load Dataset
    # --------------------------
    dataset = MultiDomainDataset(config['dataset'])

    # Prepare data loaders (train/val/test).
    # train_loader, val_loader, test_loader = dataset.get_dataloaders()
    train_loader = dataset.get_train_dataset(domain_idx=0)["data"]
    val_loader = dataset.get_val_dataset(domain_idx=0)["data"]
    test_loader = dataset.get_test_dataset(domain_idx=0)["data"]

    # --------------------------
    # 3) Select Model
    # --------------------------
    model_name = config['model']['name']
    deep_ctr_lora_list = [
        'mlp_lora', 'wdl_lora', 'nfm_lora', 'autoint_lora',
        'deepfm_lora', 'dcn_lora', 'xdeepfm_lora', 'fibinet_lora', 'pnn_lora'
    ]
    deep_ctr_list = [
        'mlp_single', 'wdl_single', 'nfm_single', 'autoint_single',
        'deepfm_single', 'dcn_single', 'xdeepfm_single', 'fibinet_single', 'pnn_single'
    ]

    if 'mlora' in model_name:
        # Custom PyTorch MLoRA class
        # model = MLoRA(dataset, config)
        print("Not implemented yet")
    elif 'star' in model_name:
        # Custom PyTorch STAR class
        # model = Star(dataset, config)
        print("Not implemented yet")
    elif in_name_list(model_name, deep_ctr_list):
        # Single domain deepctr-torch approach
        model = DeepCTR_Torch_Wrapper(dataset, config)
    elif in_name_list(model_name, deep_ctr_lora_list):
        # You would implement a variant with "LoRA" ideas integrated
        # or adapt your MLoRA code to the deepctr-torch structure
        model = MLoRA(dataset, config)  # or a separate "DeepCTR_LORA"
        print("Not implemented yet")
    else:
        print("model: {} not found".format(model_name))
        return

    # --------------------------
    # 4) (Optional) FLOPs
    # --------------------------
    # In PyTorch, you can use packages like ptflops or fvcore to estimate FLOPs
    # Example with ptflops (pip install ptflops):
    # from ptflops import get_model_complexity_info
    # dummy_input = (batch_size, number_of_features)
    # macs, params = get_model_complexity_info(model, dummy_input, as_strings=True)
    # print("FLOPS (MACs):", macs, "Params:", params)

    # --------------------------
    # 5) Train Model
    # --------------------------
    # For deepctr-torch wrappers, we can compile + call a custom train loop
    if hasattr(model, 'compile'):
        # Example: model is a DeepCTR_Torch_Wrapper
        model.compile(
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
            loss=nn.BCEWithLogitsLoss()
        )
        model.train_model(train_loader, epochs=config.get('train_epochs', 5))
    else:
        # If you have your own .train() method in MLoRA or Star
        model.train_model(train_loader, epochs=config.get('train_epochs', 5))

    # --------------------------
    # 6) Evaluate on Test
    # --------------------------
    if hasattr(model, 'evaluate_model'):
        avg_loss, avg_auc = model.evaluate_model(test_loader)
        domain_loss, domain_auc = None, None  # adapt if multi-domain
    else:
        # If your custom class uses a different interface
        avg_loss, avg_auc, domain_loss, domain_auc = model.val_and_test("test")

    print("Test Result:")
    print("Avg Loss:", avg_loss, "AUC:", avg_auc)

    # --------------------------
    # 7) (Optional) PyTorch Quantization
    # --------------------------
    # TFLite is TensorFlow-specific; for PyTorch you can do dynamic quantization:
    #
    # from torch.quantization import quantize_dynamic
    # # Suppose we only quantize Linear layers
    # q_model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    #
    # # Save model
    # quantized_model_path = "model_quantized.pth"
    # torch.save(q_model.state_dict(), quantized_model_path)
    # print("Quantized model saved to:", quantized_model_path)
    #
    # # Check file size
    # file_size_bytes = os.path.getsize(quantized_model_path)
    # print("Quantized model file size: {:.2f} MB".format(file_size_bytes / (1024 * 1024)))

    # If you still want TFLite, you’d typically export your model to ONNX, then convert:
    #   1) torch.onnx.export(...)
    #   2) Use onnx2tf or TF-onnx to produce a .tflite

    # --------------------------
    # 8) (Optional) Freeze / Partial Fine-Tuning
    # --------------------------
    # For example, in PyTorch you can freeze part of the network:
    if "freeze" in config['model']['dense']:
        # 1) Reload from checkpoint if you have it
        #    or in PyTorch you'd do:
        # pretrained_dict = torch.load('your_checkpoint.pt')
        # model.load_state_dict(pretrained_dict)
        #
        # 2) Freeze layers based on name matching
        for name, param in model.named_parameters():
            if layer_matching(name, "some_pattern"):
                param.requires_grad = False

        # Then re-run training or partial training
        # model.train_model(train_loader, epochs=...)
        pass

    # --------------------------
    # 9) Save or return results
    # --------------------------
    # Suppose you have a custom method
    if hasattr(model, 'save_result'):
        model.save_result(avg_loss, avg_auc, domain_loss, domain_auc)

    return avg_loss, avg_auc, domain_loss, domain_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Train config file", required=False,
                        default="config/Movielens/autoint_mlora.json")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    main(config)
