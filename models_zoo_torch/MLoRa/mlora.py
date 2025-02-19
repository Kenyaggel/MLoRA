import os
import os.path as osp
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score  # to replace your custom AUC
from .partitioned_norm import PartitionedNorm  # your PyTorch version
from .mlora_fcn import MLoRAFCN
from .mlora_fcn_pretrain import MLoRAFCN as MLoRAFCN_PRE
# freeze W,训练lora
from .mlora_fcn_freeze import MLoRAFCN as MLoRAFCN_FREEZE

from ..deepctr_torch.models.basemodel import BaseModel  # your custom PyTorch base
# or define your own base if you prefer

class MLoRA(BaseModel):
    def __init__(self, dataset, config):
        super(MLoRA, self).__init__(dataset, config)
        self.dataset = dataset
        self.model_config = config['model']
        self.train_config = config['train']
        self.n_domain = self.dataset.n_domain
        self.n_uid = self.dataset.n_uid
        self.n_pid = self.dataset.n_pid

        # e.g.:
        # self.checkpoint_path = ...
        # self.early_stop_step = ...
        # self.finetune_train_sequence = ...
        # (All from your original code, but adapt as needed.)

        # Build the PyTorch model
        self.model = self.build_model_structure()

        # Setup optimizer, loss, etc. (replaces model.compile)
        if self.train_config.get('optimizer', 'adam') == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.train_config['learning_rate'])
        else:
            raise ValueError("Only 'adam' is demonstrated here.")

        # BCE loss for binary classification
        self.criterion = nn.BCELoss()

    def build_model_structure(self):
        """
        Build the entire PyTorch model (embedding layers, FCs, LoRA blocks, etc.)
        Return an nn.Module that we'll call self.model.
        """
        return MLoRAModel(self.n_domain, self.n_uid, self.n_pid, self.model_config, self.train_config, self.dataset)

    def train_model(self):
        """
        Replaces your old .train() method that used model.fit().
        We'll do a standard PyTorch epoch/batch loop.
        """
        # Possibly load from checkpoint if needed
        if self.train_config.get('train_from_ckp', False):
            ckp_path = ...  # define your checkpoint path
            print("Loading checkpoint from:", ckp_path)
            self.load_model(ckp_path)

        # Example domain sequence
        train_sequence = list(range(self.n_domain))

        # If "finetune" in self.model_config['dense'] logic
        if "finetune" in self.model_config['dense']:
            # replicate your logic around random_delete, etc.
            if self.model_config['pretrain_judge'] == 'True':
                train_sequence = self.random_delete(train_sequence)
            else:
                train_sequence = self.finetune_train_sequence

        # Possibly an early val check
        if 'freeze' in self.model_config['dense']:
            if self.model_config['pretrain_judge'] == "False":
                avg_loss, avg_auc, domain_loss, domain_auc = self.val_and_test("val")
                self.early_stop_step(avg_auc)

        # Actual training loop
        n_epochs = self.train_config['epoch']
        for epoch in range(n_epochs):
            print(f"Epoch: {epoch} {'-'*30}")
            random.shuffle(train_sequence)

            epoch_start_time = time.time()
            self.model.train()  # set to training mode

            for d_idx in train_sequence:
                # Retrieve domain’s training loader
                train_info = self.dataset.train_dataset[d_idx]  # -> {"data": DataLoader, ...}
                train_loader = train_info['data']

                for batch_features, batch_labels in train_loader:
                    # batch_features is dict-like or a single tensor, depending on your dataset
                    # parse them for user, item, domain
                    user = batch_features['uid'].long().squeeze(-1)   # shape [B]
                    item = batch_features['pid'].long().squeeze(-1)   # shape [B]
                    domain = batch_features['domain'].long().squeeze(-1)  # shape [B]
                    # label
                    labels = batch_labels.float().squeeze(-1)  # shape [B]

                    # forward
                    preds = self.model(user, item, domain)

                    # compute loss
                    loss = self.criterion(preds, labels)

                    # backward
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            print("Training time (this epoch):", time.time() - epoch_start_time)

            # Validation
            print("Val Result:")
            avg_loss, avg_auc, domain_loss, domain_auc = self.val_and_test("val")
            self.epoch = epoch
            if self.early_stop_step(avg_auc):
                break

            # Test
            print("Test Result:")
            test_avg_loss, test_avg_auc, domain_loss, domain_auc = self.val_and_test("test")

    def val_and_test(self, split="val"):
        """
        Evaluate on each domain’s val or test DataLoader, compute average metrics.
        Return (avg_loss, avg_auc, domain_loss, domain_auc)
        """
        self.model.eval()
        domain_losses = []
        domain_aucs = []
        domain_sizes = []
        with torch.no_grad():
            for d_idx in range(self.n_domain):
                if split == "val":
                    loader_info = self.dataset.val_dataset[d_idx]
                else:
                    loader_info = self.dataset.test_dataset[d_idx]

                data_loader = loader_info['data']
                all_preds = []
                all_labels = []
                total_loss = 0.0
                for batch_features, batch_labels in data_loader:
                    user = batch_features['uid'].long().squeeze(-1)
                    item = batch_features['pid'].long().squeeze(-1)
                    domain = batch_features['domain'].long().squeeze(-1)
                    labels = batch_labels.float().squeeze(-1)

                    preds = self.model(user, item, domain)
                    loss = self.criterion(preds, labels)
                    total_loss += loss.item() * len(labels)

                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())

                all_preds = np.concatenate(all_preds)
                all_labels = np.concatenate(all_labels)
                domain_auc = roc_auc_score(all_labels, all_preds)
                domain_loss = total_loss / len(all_labels)

                domain_losses.append(domain_loss)
                domain_aucs.append(domain_auc)
                domain_sizes.append(len(all_labels))

        # Weighted average (by domain size) or simple average
        avg_loss = np.average(domain_losses, weights=domain_sizes)
        avg_auc = np.average(domain_aucs, weights=domain_sizes)
        return avg_loss, avg_auc, domain_losses, domain_aucs

    def load_model(self, checkpoint_path):
        """
        In PyTorch, to load a model from a checkpoint:
        """
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(state_dict)

    def save_model(self, checkpoint_path):
        """
        Save model state dict to file.
        """
        torch.save(self.model.state_dict(), checkpoint_path)

    def random_delete(self, train_sequence):
        # Same logic as your TF code
        a = random.randint(0, self.n_domain - 1)
        self.finetune_train_sequence = [train_sequence[a]]
        train_sequence.pop(a)
        b = random.randint(0, self.n_domain - 2)
        self.finetune_train_sequence.append(train_sequence[b])
        train_sequence.pop(b)
        return train_sequence


class MLoRAModel(nn.Module):
    def __init__(self, n_domain, n_uid, n_pid, model_config, train_config, dataset):
        super(MLoRAModel, self).__init__()
        self.n_domain = n_domain
        self.n_uid = n_uid
        self.n_pid = n_pid
        self.model_config = model_config
        self.train_config = train_config
        self.dataset = dataset

        # Build embeddings
        self.user_dim = model_config['user_dim']
        self.item_dim = model_config['item_dim']
        self.domain_dim = model_config['domain_dim']

        # user embedding
        self.user_emb = nn.Embedding(self.n_uid, self.user_dim)
        # If you need pre-trained embeddings, you can load them from dataset
        if self.train_config['load_pretrain_emb']:
            embedding_matrix = self._load_pretrain_embedding(dataset.user_emb, self.n_uid, self.user_dim)
            self.user_emb.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.user_emb.weight.requires_grad = self.train_config['emb_trainable']

        # item embedding
        self.item_emb = nn.Embedding(self.n_pid, self.item_dim)
        if self.train_config['load_pretrain_emb']:
            embedding_matrix = self._load_pretrain_embedding(dataset.item_emb, self.n_pid, self.item_dim)
            self.item_emb.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.item_emb.weight.requires_grad = self.train_config['emb_trainable']

        # domain embedding
        self.domain_emb = nn.Embedding(self.n_domain, self.domain_dim)
        # Freeze domain embedding if needed:
        if 'freeze' in model_config['dense'] and model_config['pretrain_judge'] == 'False':
            self.domain_emb.weight.requires_grad = False

        # PartitionedNorm or BN
        if model_config['norm'] == 'pn':
            # If freeze and pretrain_judge == False -> not trainable
            trainable_pn = True
            if 'freeze' in model_config['dense'] and model_config['pretrain_judge'] == 'False':
                trainable_pn = False
            self.partitioned_norm = PartitionedNorm(n_domain, trainable=trainable_pn)
        elif model_config['norm'] == 'bn':
            self.batch_norm = nn.BatchNorm1d(model_config['user_dim']+model_config['item_dim']+model_config['domain_dim'])
        else:
            self.partitioned_norm = None
            self.batch_norm = None

        # Build the hidden layers
        self.hidden_layers = nn.ModuleList()
        dense_type = model_config['dense']
        hidden_dim_list = model_config['hidden_dim']
        lora_r = model_config.get('lora_r', 8)
        lora_reduce = model_config.get('lora_reduce', 4)
        dropout_rate = model_config.get('dropout', 0.0)

        for i, h_dim in enumerate(hidden_dim_list):
            if dense_type == 'dense':
                layer = nn.Linear(-1, h_dim)  # placeholder, see below
            elif dense_type == 'mlora':
                layer = MLoRAFCN(n_domain, h_dim, activation='relu',
                                 lora_r=lora_r, lora_reduce=lora_reduce,
                                 dropout_rate=dropout_rate,
                                 is_finetune=train_config['train_from_ckp'],
                                 trainable=True)
            elif 'mlora_freeze' in dense_type:
                if model_config['pretrain_judge'] == 'True':
                    layer = MLoRAFCN_PRE(n_domain, h_dim, ...)
                else:
                    layer = MLoRAFCN_FREEZE(n_domain, h_dim, ...)
            else:
                layer = nn.Linear(-1, h_dim)
            self.hidden_layers.append(layer)

        # Final output layer (freeze if needed)
        self.out_layer = nn.Linear(hidden_dim_list[-1] if hidden_dim_list else (self.user_dim+self.item_dim+self.domain_dim),
                                   1)
        if 'freeze' in model_config['dense'] and model_config['pretrain_judge'] == 'False':
            for param in self.out_layer.parameters():
                param.requires_grad = False

    def _load_pretrain_embedding(self, emb_dict, n, dim):
        """Convert your dataset’s string embeddings to a numpy array."""
        embedding_matrix = np.zeros((n, dim), dtype=np.float32)
        for k, v in emb_dict.items():
            idx = int(k)
            vals = np.fromstring(v, sep=' ', dtype=np.float32)
            embedding_matrix[idx] = vals
        return embedding_matrix

    def forward(self, user_idx, item_idx, domain_idx):
        """
        user_idx: shape [B]
        item_idx: shape [B]
        domain_idx: shape [B]
        """
        # 1) embeddings
        u = self.user_emb(user_idx)     # [B, user_dim]
        i = self.item_emb(item_idx)     # [B, item_dim]
        d = self.domain_emb(domain_idx) # [B, domain_dim]

        x = torch.cat([u, i, d], dim=1) # [B, user_dim+item_dim+domain_dim]

        # 2) Norm
        if self.partitioned_norm is not None:
            # partitioned norm needs domain_idx also
            x = self.partitioned_norm(x, domain_idx)
        elif self.batch_norm is not None:
            x = self.batch_norm(x)

        # 3) Pass through hidden layers
        for layer in self.hidden_layers:
            # Depending on your design, if it is an MLoRAFCN,
            # you might pass both x and domain_idx
            if isinstance(layer, (MLoRAFCN, MLoRAFCN_PRE, MLoRAFCN_FREEZE)):
                x = layer(x, domain_idx)  # custom
            else:
                x = layer(x)
                x = F.relu(x)

        # 4) Final output layer
        logit = self.out_layer(x)
        out = torch.sigmoid(logit).squeeze(-1)  # shape [B]
        return out
