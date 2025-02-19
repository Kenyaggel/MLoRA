import os
import os.path as osp
import time
import json
import datetime

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score

class BaseModel(object):
    def __init__(self, dataset, config):
        """
        dataset: an object containing self.n_uid, self.n_pid, self.n_domain,
                 plus domain-specific data loaders for train/val/test.
        config: a dict with 'model', 'train', etc.
        """
        self.n_uid = dataset.n_uid
        self.n_pid = dataset.n_pid
        self.n_domain = dataset.n_domain
        self.dataset = dataset
        self.config = config
        self.model_config = config['model']
        self.train_config = config['train']

        # Checkpoint folder
        timestamp = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime())
        self.checkpoint_path = osp.join(
            self.train_config['checkpoint_path'],
            self.model_config['name'],
            self.dataset.conf['name'],
            self.dataset.conf['domain_split_path'],
            timestamp,
            "model_parameters.pt"  # use .pt for PyTorch
        )
        # Result folder
        self.result_path = osp.join(
            self.train_config['result_save_path'],
            self.model_config['name'],
            self.dataset.conf['name'],
            self.dataset.conf['domain_split_path']
        )

        # Build PyTorch model (nn.Module)
        self.model = self.build_model()

        # For domain-splitting logic
        self.finetune_train_sequence = None

        # Early stop
        self._build_early_stop()

        # Track current epoch
        self.epoch = 0

    def build_model(self):
        """
        Must return a PyTorch nn.Module
        """
        raise NotImplementedError("You must implement build_model() in subclass")

    def train(self):
        """
        Subclasses implement their training loop or call helper methods.
        """
        raise NotImplementedError("You must implement train() in subclass")

    # -------------------------------------------------------------------------
    # Domain-by-domain training (optional, depending on your TF code usage)
    # -------------------------------------------------------------------------
    def separate_train_val_test(self, init_params=True):
        """
        Example replication of your "separate_train_val_test" approach, where
        we train the model on each domain individually, then evaluate on val/test.
        If init_params=True, we re-initialize or reset model weights each domain.
        """
        # In TensorFlow, you called "graph._unsafe_unfinalize()".
        # In PyTorch, there's no "graph" concept. We just proceed.

        # domain_loss / domain_auc
        domain_loss = {}
        domain_auc = {}
        all_loss = 0.0
        all_auc = 0.0

        # Optionally re-initialize weights or store a snapshot of initial weights.
        # For a typical PyTorch flow, once the model is created, we have random init.
        # If you want to store it, do something like:
        init_state_dict = self.model.state_dict()
        init_state_dict_clone = {
            k: v.clone() for k, v in init_state_dict.items()
        }

        # We'll define a simpler approach: for domain_idx in train_dataset, do training
        for domain_idx, train_d in self.dataset.train_dataset.items():
            # Possibly re-load initial weights if init_params is True
            if init_params:
                self.model.load_state_dict(init_state_dict_clone)

            # Build an optimizer. In TF code, you switched to GradientDescent:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.train_config['learning_rate'])
            criterion = nn.BCELoss()

            # Train on domain domain_idx
            print(f"[separate_train_val_test] Train on domain: {domain_idx}")
            self._train_one_domain(
                domain_idx=domain_idx,
                optimizer=optimizer,
                criterion=criterion,
                epochs=self.train_config['epoch'],
                patience=self.train_config['patience']
            )

            # Evaluate on test
            test_info = self.dataset.test_dataset[domain_idx]
            test_loss, test_auc = self._eval_one_domain(domain_idx, criterion, split="test")
            domain_loss[domain_idx] = test_loss
            domain_auc[domain_idx] = test_auc
            all_loss += test_loss
            all_auc += test_auc

        # Optionally restore meta-weight
        self.model.load_state_dict(init_state_dict_clone)

        avg_loss = all_loss / len(domain_loss)
        avg_auc = all_auc / len(domain_auc)
        print("Loss: ", domain_loss)
        self._format_print_domain_metric("AUC", domain_auc)
        weighted_auc = self._weighted_auc("test", domain_auc)
        print(f"Overall test Loss: {avg_loss}, AUC: {avg_auc}, Weighted AUC: {weighted_auc}")
        return avg_loss, avg_auc, domain_loss, domain_auc

    def _train_one_domain(self, domain_idx, optimizer, criterion, epochs, patience):
        """
        Simple example: train domain_idx's data for up to `epochs`, with early stopping on val AUC.
        Note: Replaces the Keras .fit() approach.
        """
        best_auc = None
        counter = 0

        train_info = self.dataset.train_dataset[domain_idx]
        val_info = self.dataset.val_dataset[domain_idx]

        # Extract data loaders
        train_loader = train_info['data']
        val_loader = val_info['data']

        for ep in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            all_preds = []
            all_labels = []

            for batch_features, batch_labels in train_loader:
                # zero grad
                optimizer.zero_grad()

                # forward
                preds = self._forward_pass(batch_features)
                labels = batch_labels.float().view(-1)
                loss = criterion(preds, labels)

                # backward
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(labels)
                # collect for AUC
                all_preds.append(preds.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())

            # compute epoch metrics
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            epoch_auc = roc_auc_score(all_labels, all_preds)
            epoch_loss = epoch_loss / len(all_labels)
            # Evaluate on val
            val_loss, val_auc = self._eval_one_domain(domain_idx, criterion, split="val")

            # Early stopping
            if best_auc is None or val_auc > best_auc:
                best_auc = val_auc
                counter = 0
                # Save checkpoint for best val
                self.save_model(self.checkpoint_path)
            else:
                counter += 1
                if counter >= patience:
                    print(f"EarlyStopping triggered at epoch={ep}, best AUC={best_auc}")
                    # reload best
                    self.load_model(self.checkpoint_path)
                    break

    def _eval_one_domain(self, domain_idx, criterion, split="val"):
        """
        Evaluate model on domain_idx's data. Return (loss, auc).
        """
        if split == "val":
            d_info = self.dataset.val_dataset[domain_idx]
        else:
            d_info = self.dataset.test_dataset[domain_idx]

        data_loader = d_info['data']
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_features, batch_labels in data_loader:
                preds = self._forward_pass(batch_features)
                labels = batch_labels.float().view(-1)
                loss = criterion(preds, labels)
                total_loss += loss.item() * len(labels)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        total_loss = total_loss / len(all_labels)
        auc_val = roc_auc_score(all_labels, all_preds)
        return total_loss, auc_val

    def _forward_pass(self, batch_features):
        """
        Convert batch_features (dict or otherwise) to model inputs, do forward pass.
        In your code, you have user, item, domain, etc.  Adjust as needed.
        """
        # Suppose batch_features is a dict: { 'uid': ..., 'pid': ..., 'domain': ... }
        # e.g.:
        user = batch_features['uid'].long().view(-1)
        item = batch_features['pid'].long().view(-1)
        domain = batch_features['domain'].long().view(-1)

        # forward
        logits = self.model(user, item, domain)  # or however your model is called
        return logits  # shape [B], use sigmoid later or model might do it

    # -------------------------------------------------------------------------
    # Val/Test for entire model across domains
    # -------------------------------------------------------------------------
    def val_and_test(self, mode):
        """
        Evaluate the model on all domains for either "val" or "test".
        In TF code, you load best weights if mode="test". We'll do that here too.
        """
        if mode == "test":
            self.load_model(self.checkpoint_path)  # load best from current run

        # domain-wise metrics
        domain_loss = {}
        domain_auc = {}
        all_loss = 0.0
        all_auc = 0.0

        if mode == "val":
            dataset_dict = self.dataset.val_dataset
        elif mode == "test":
            dataset_dict = self.dataset.test_dataset
        else:
            raise ValueError(f"Mode must be 'val' or 'test', got {mode}")

        # use a criterion for loss
        criterion = nn.BCELoss()

        for d_idx, d_info in dataset_dict.items():
            data_loader = d_info['data']
            # Evaluate
            t_loss, t_auc = self._eval_one_domain(d_idx, criterion, split=mode)
            domain_loss[d_idx] = t_loss
            domain_auc[d_idx] = t_auc
            all_loss += t_loss
            all_auc += t_auc

        avg_loss = all_loss / len(domain_loss)
        avg_auc = all_auc / len(domain_auc)

        print("Loss:", domain_loss)
        self._format_print_domain_metric("AUC", domain_auc)
        weighted_auc = self._weighted_auc(mode, domain_auc)
        print(f"time: {datetime.datetime.now()}")
        print(f"Overall {mode} Loss: {avg_loss}, AUC: {avg_auc}, Weighted AUC: {weighted_auc}")
        return avg_loss, avg_auc, domain_loss, domain_auc

    def val_and_test_total(self, mode):
        """
        If your dataset has a combined 'total_val_dataset' or 'total_test_dataset',
        you can evaluate the entire set at once. The TF code calls model.evaluate(...).
        We'll do a manual loop in PyTorch for that combined dataset.
        """
        if mode == "test":
            self.load_model(self.checkpoint_path)

        if mode == "val":
            dataset_info = self.dataset.total_val_dataset
        elif mode == "test":
            dataset_info = self.dataset.total_test_dataset
        else:
            raise ValueError(f"Mode must be 'val' or 'test', got {mode}")

        data_loader = dataset_info['data']
        n_step = dataset_info['n_step']

        criterion = nn.BCELoss()

        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_labels in data_loader:
                preds = self._forward_pass(batch_features)
                labels = batch_labels.float().view(-1)
                loss = criterion(preds, labels)
                total_loss += loss.item() * len(labels)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        total_loss = total_loss / len(all_labels)
        auc_val = roc_auc_score(all_labels, all_preds)

        print(f"Overall {mode} Loss: {total_loss}, AUC: {auc_val}")
        return total_loss, auc_val

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    def _format_print_domain_metric(self, name, domain_metric):
        print(f"{name}:")
        for key, value in domain_metric.items():
            print(f"  Domain {key}: {value}")

    def _weighted_auc(self, mode, domain_auc):
        """
        Weighted average AUC based on domain data size.
        dataset_info = self.dataset.dataset_info
        domain_auc is a dict: domain_idx -> AUC
        """
        data_info = self.dataset.dataset_info
        if "val" in mode:
            tag = "n_val"
        elif "test" in mode:
            tag = "n_test"
        else:
            tag = "n_train"

        weighted_auc = 0.0
        total_num = 0
        for domain_idx, auc_val in domain_auc.items():
            domain_size = data_info[domain_idx][tag]
            weighted_auc += domain_size * auc_val
            total_num += domain_size

        if total_num == 0:
            return 0.0
        return weighted_auc / total_num

    def save_model(self, path):
        """
        Save PyTorch model state dict.
        """
        folder = os.path.dirname(path)
        if not osp.exists(folder):
            os.makedirs(folder)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """
        Load PyTorch model state dict.
        """
        if osp.exists(path):
            self.model.load_state_dict(torch.load(path, map_location='cpu'))
            print(f"Loaded model weights from {path}")
        else:
            print(f"No checkpoint found at {path} (skipping load)")

    def save_result(self, avg_loss, avg_auc, domain_loss=-1, domain_auc=-1):
        """
        Save results + model in a subfolder.
        Mimics your original 'save_result(...)' method.
        """
        avg_loss, avg_auc = float(avg_loss), float(avg_auc)
        result_folder_name = "loss_{:.3f}_auc_{:.3f}_{}".format(
            avg_loss, avg_auc, time.strftime("%a-%b-%d-%H-%M-%S", time.localtime())
        )
        result_path = osp.join(self.result_path, result_folder_name)
        if not osp.exists(result_path):
            os.makedirs(result_path)

        # Save dataset_info, config, result
        with open(osp.join(result_path, "dataset_info.json"), 'w') as f:
            json.dump(self.dataset.dataset_info, f, indent=2)
        with open(osp.join(result_path, "config.json.example"), 'w') as f:
            json.dump(self.config, f, indent=2)
        with open(osp.join(result_path, "result.json"), 'w') as f:
            json.dump({
                "avg_loss": avg_loss,
                "avg_auc": avg_auc,
                "domain_loss": domain_loss,
                "domain_auc": domain_auc
            }, f, indent=2)

        # Save the model
        self.save_model(osp.join(result_path, "model_parameters.pt"))

    # -------------------------------------------------------------------------
    # Early Stop Logic
    # -------------------------------------------------------------------------
    def _build_early_stop(self):
        self.patience = self.train_config.get('patience', 5)
        self.counter = 0
        self.best_metric = None
        self.early_stop = False

    def early_stop_step(self, metric):
        """
        If the AUC doesn't improve, increment self.counter.
        If it surpasses patience, set self.early_stop = True.
        Otherwise, reset counter and save checkpoint as best.
        """
        folder = os.path.dirname(self.checkpoint_path)
        if not osp.exists(folder):
            os.makedirs(folder)

        if self.best_metric is None:
            self.best_metric = metric
            self.save_model(self.checkpoint_path)
        elif metric <= self.best_metric:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}, Best AUC: {self.best_metric}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_model(self.checkpoint_path)
            self.best_metric = metric
            self.counter = 0
        return self.early_stop
