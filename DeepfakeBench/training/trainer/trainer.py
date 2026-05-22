# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: trainer
import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import pickle
import datetime
import logging
import numpy as np
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from metrics.base_metrics_class import Recorder
from torch.optim.swa_utils import AveragedModel, SWALR
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn import metrics
from metrics.utils import get_test_metrics

from optimizor.sam import SAM
from optimizor.pcgrad import PCGrad

FFpp_pool=['FaceForensics++','FF-DF','FF-F2F','FF-FS','FF-NT']#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Asymmetric Mixup ──────────────────────────────────────────────────────────
def asymmetric_mixup(x, y, alpha=1.0, gamma=5.0):
    """
    Asymmetric Mixup: Real-Real / Fake-Fake → standard mixup label;
    Real-Fake → y_mixed = 1 - (real_prop ** gamma)  (aggressively Fake).
    y=0 → Real, y=1 → Fake.
    Returns mixed images and soft labels in [0, 1].
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y.float(), y[index].float()
    lam_t = torch.tensor(lam, dtype=torch.float32, device=x.device)
    # proportion of the Fake image in the blend
    lam_fake = torch.where(y_a == 1.0, lam_t, 1.0 - lam_t)
    mixed_y_std  = lam * y_a + (1 - lam) * y_b               # same-class pairs
    mixed_y_asym = 1.0 - (1.0 - lam_fake) ** gamma            # cross-class pairs
    mixed_y = torch.where(y_a == y_b, mixed_y_std, mixed_y_asym)
    return mixed_x, mixed_y


def hardest_k_mixup(model, data_dict, K, alpha=1.0, gamma=5.0, selection='hardest'):
    """
    Built on top of asymmetric_mixup's randperm structure.
    K-candidate replaces only real+fake (y_a=0, y_b=1) pairs.
    real+real / fake+fake / fake+real pairs use shared-λ randperm (same as asymmetric_mixup).
    """
    x, y = data_dict['image'], data_dict['label']
    real_idx = (y == 0).nonzero(as_tuple=True)[0]   # [R]
    fake_idx = (y == 1).nonzero(as_tuple=True)[0]   # [F]
    B = x.size(0)
    R, F_orig = len(real_idx), len(fake_idx)

    if K <= 1 or R == 0 or F_orig == 0:
        mixed_x, label_soft = asymmetric_mixup(x, y, alpha, gamma)
        return {**data_dict, 'image': mixed_x, 'label_soft': label_soft}

    # ── 1. Randperm base (shared λ, same structure as asymmetric_mixup) ────
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    lam_t = torch.tensor(lam, dtype=torch.float32, device=x.device)
    index = torch.randperm(B, device=x.device)

    # Pairing-type masks (anchor → partner)
    rr = (y == 0) & (y[index] == 0)   # real+real
    ff = (y == 1) & (y[index] == 1)   # fake+fake
    rf = (y == 0) & (y[index] == 1)   # real+fake → K-candidate
    fr = (y == 1) & (y[index] == 0)   # fake+real → asymmetric

    # ── 2. Real+real (shared λ, randperm partner) ─────────────────────────
    rr_x = lam_t * x[rr] + (1.0 - lam_t) * x[index[rr]]
    rr_soft = torch.zeros(rr.sum().item(), device=x.device)

    # ── 3. Fake+fake (shared λ, randperm partner) ─────────────────────────
    n_ff = ff.sum().item()
    if n_ff > 0:
        ff_x = lam_t * x[ff] + (1.0 - lam_t) * x[index[ff]]
        ff_soft = torch.ones(n_ff, device=x.device)
    else:
        ff_x = torch.empty(0, *x.shape[1:], device=x.device)
        ff_soft = torch.empty(0, device=x.device)

    # ── 4. Fake+real (shared λ, randperm partner, asymmetric label) ───────
    fr_idx = fr.nonzero(as_tuple=True)[0]
    n_fr = len(fr_idx)
    if n_fr > 0:
        fr_x = lam_t * x[fr_idx] + (1.0 - lam_t) * x[index[fr_idx]]
        fr_soft = (1.0 - ((1.0 - lam_t) ** gamma)).expand(n_fr)
    else:
        fr_x = torch.empty(0, *x.shape[1:], device=x.device)
        fr_soft = torch.empty(0, device=x.device)

    # ── 5. Real+fake K-candidate (replaces randperm rf pairs) ──────────────
    rf_idx = rf.nonzero(as_tuple=True)[0]
    n_rf = len(rf_idx)

    if n_rf == 0:
        parts_x = [x for x in [rr_x, ff_x, fr_x] if x is not None and x.numel() > 0]
        parts_soft = [s for s in [rr_soft, ff_soft, fr_soft] if s is not None and s.numel() > 0]
        parts_label = [
            torch.zeros(rr.sum().item(), dtype=y.dtype, device=x.device),
            torch.ones(n_ff, dtype=y.dtype, device=x.device) if n_ff > 0 else torch.empty(0, dtype=y.dtype, device=x.device),
            y[fr_idx] if n_fr > 0 else torch.empty(0, dtype=y.dtype, device=x.device),
        ]
        new_x = torch.cat(parts_x, dim=0)
        new_label_soft = torch.cat(parts_soft, dim=0)
        new_label = torch.cat(parts_label, dim=0)
        return {**data_dict, 'image': new_x, 'label': new_label, 'label_soft': new_label_soft}

    K_eff = min(K, F_orig)

    # K distinct fakes per rf-real (without replacement)
    cand_fake = torch.stack([
        fake_idx[torch.randperm(F_orig, device=x.device)[:K_eff]]
        for _ in range(n_rf)
    ], dim=1)                                                             # [K_eff, n_rf]

    # Per-(k,r) independent λ
    lam_kr = np.random.beta(alpha, alpha, size=(K_eff, n_rf)) if alpha > 0 else np.ones((K_eff, n_rf))
    lam_t_kr = x.new_tensor(lam_kr).float()                               # [K_eff, n_rf]
    soft_val_kr = 1.0 - (lam_t_kr ** gamma)                                # [K_eff, n_rf]

    # Build K_eff * n_rf mixed images
    x_real_rep = (x[rf_idx]
                  .unsqueeze(0)
                  .expand(K_eff, -1, -1, -1, -1)
                  .reshape(K_eff * n_rf, *x.shape[1:]))
    x_fake_rep = x[cand_fake.reshape(-1)]
    lam_exp = lam_t_kr.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *x.shape[1:])
    lam_exp = lam_exp.reshape(K_eff * n_rf, *x.shape[1:])
    mixed_kr = lam_exp * x_real_rep + (1.0 - lam_exp) * x_fake_rep
    soft_val_exp = soft_val_kr.reshape(-1)                                 # [K_eff * n_rf]

    if selection == 'mean':
        new_x = torch.cat([rr_x, ff_x, fr_x, mixed_kr], dim=0)
        new_label_soft = torch.cat([rr_soft, ff_soft, fr_soft, soft_val_exp], dim=0)
        new_label = torch.cat([
            torch.zeros(rr.sum().item(), dtype=y.dtype, device=x.device),
            torch.ones(n_ff, dtype=y.dtype, device=x.device) if n_ff > 0 else torch.empty(0, dtype=y.dtype, device=x.device),
            y[fr_idx] if n_fr > 0 else torch.empty(0, dtype=y.dtype, device=x.device),
            torch.zeros(K_eff * n_rf, dtype=y.dtype, device=x.device),
        ], dim=0)
        return {**data_dict, 'image': new_x, 'label': new_label,
                'label_soft': new_label_soft, 'n_rf': n_rf,
                'mixup_k': K_eff, 'mixup_selection': 'mean'}

    # ── Select candidate per rf-real (random or hardest) ──────────────────
    if selection == 'random':
        best_k = torch.randint(0, K_eff, (n_rf,), device=x.device)
    else:  # hardest
        model_module = model.module if hasattr(model, 'module') else model
        with torch.no_grad():
            feat_kr = model_module.features({**data_dict, 'image': mixed_kr})
            pred_kr = model_module.classifier(feat_kr)
        log_p = F.log_softmax(pred_kr, dim=1)
        loss_kr = -(soft_val_exp * log_p[:, 1] +
                    (1.0 - soft_val_exp) * log_p[:, 0])
        best_k = loss_kr.view(K_eff, n_rf).argmax(dim=0)

    flat_idx = best_k * n_rf + torch.arange(n_rf, device=x.device)
    rf_x = mixed_kr[flat_idx]
    rf_soft = soft_val_exp[flat_idx]

    # ── 6. Combine all ────────────────────────────────────────────────────
    new_x = torch.cat([rr_x, ff_x, fr_x, rf_x], dim=0)
    new_label_soft = torch.cat([rr_soft, ff_soft, fr_soft, rf_soft], dim=0)
    new_label = torch.cat([
        torch.zeros(rr.sum().item(), dtype=y.dtype, device=x.device),
        torch.ones(n_ff, dtype=y.dtype, device=x.device) if n_ff > 0 else torch.empty(0, dtype=y.dtype, device=x.device),
        y[fr_idx] if n_fr > 0 else torch.empty(0, dtype=y.dtype, device=x.device),
        torch.zeros(n_rf, dtype=y.dtype, device=x.device),
    ], dim=0)

    return {**data_dict, 'image': new_x, 'label': new_label,
            'label_soft': new_label_soft}
# ─────────────────────────────────────────────────────────────────────────────


class Trainer(object):
    def __init__(
        self,
        config,
        model,
        optimizer,
        scheduler,
        logger,
        metric_scoring='auc',
        time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
        swa_model=None
        ):
        # check if all the necessary components are implemented
        if config is None or model is None or optimizer is None or logger is None:
            raise ValueError("config, model, optimizier, logger, and tensorboard writer must be implemented")

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.swa_model = swa_model
        self.writers = {}  # dict to maintain different tensorboard writers for each dataset and metric
        self.logger = logger
        self.metric_scoring = metric_scoring
        # maintain the best metric of all epochs
        self.best_metrics_all_time = defaultdict(
            lambda: defaultdict(lambda: float('-inf')
            if self.metric_scoring != 'eer' else float('inf'))
        )
        self.speed_up()  # move model to GPU

        # get current time
        self.timenow = time_now
        # create directory path
        if 'task_target' not in config:
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + '_' + self.timenow
            )
        else:
            task_str = f"_{config['task_target']}" if config['task_target'] is not None else ""
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + task_str + '_' + self.timenow
            )
        os.makedirs(self.log_dir, exist_ok=True)

    def get_writer(self, phase, dataset_key, metric_key):
        writer_key = f"{phase}-{dataset_key}-{metric_key}"
        if writer_key not in self.writers:
            # update directory path
            writer_path = os.path.join(
                self.log_dir,
                phase,
                dataset_key,
                metric_key,
                "metric_board"
            )
            os.makedirs(writer_path, exist_ok=True)
            # update writers dictionary
            self.writers[writer_key] = SummaryWriter(writer_path)
        return self.writers[writer_key]


    def speed_up(self):
        self.model.to(device)
        self.model.device = device
        if self.config['ddp'] == True:
            num_gpus = torch.cuda.device_count()
            print(f'avai gpus: {num_gpus}')
            # local_rank=[i for i in range(0,num_gpus)]
            self.model = DDP(self.model, device_ids=[self.config['local_rank']],find_unused_parameters=True, output_device=self.config['local_rank'])
            #self.optimizer =  nn.DataParallel(self.optimizer, device_ids=[int(os.environ['LOCAL_RANK'])])

    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            suffix = model_path.split('.')[-1]
            if suffix == 'p':
                self.model.load_state_dict(saved.state_dict())
            else:
                self.model.load_state_dict(saved)
            self.logger.info('Model found in {}'.format(model_path))
        else:
            raise NotImplementedError(
                "=> no model found at '{}'".format(model_path))

    def save_ckpt(self, phase, dataset_key,ckpt_info=None):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"ckpt_best.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        if self.config['ddp'] == True:
            torch.save(self.model.state_dict(), save_path)
        else:
            if 'svdd' in self.config['model_name']:
                torch.save({'R': self.model.R,
                            'c': self.model.c,
                            'state_dict': self.model.state_dict(),}, save_path)
            else:
                torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Checkpoint saved to {save_path}, current ckpt is {ckpt_info}")

    def save_swa_ckpt(self):
        save_dir = self.log_dir
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"swa.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        torch.save(self.swa_model.state_dict(), save_path)
        self.logger.info(f"SWA Checkpoint saved to {save_path}")


    def save_feat(self, phase, fea, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        features = fea
        feat_name = f"feat_best.npy"
        save_path = os.path.join(save_dir, feat_name)
        np.save(save_path, features)
        self.logger.info(f"Feature saved to {save_path}")

    def save_data_dict(self, phase, data_dict, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'data_dict_{phase}.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(data_dict, file)
        self.logger.info(f"data_dict saved to {file_path}")

    def save_metrics(self, phase, metric_one_dataset, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, 'metric_dict_best.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(metric_one_dataset, file)
        self.logger.info(f"Metrics saved to {file_path}")

    def train_step(self,data_dict):
        if self.config['optimizer']['type']=='sam':
            for i in range(2):
                predictions = self.model(data_dict)
                losses = self.model.get_losses(data_dict, predictions)
                if i == 0:
                    pred_first = predictions
                    losses_first = losses
                self.optimizer.zero_grad()
                losses['overall'].backward()
                if i == 0:
                    self.optimizer.first_step(zero_grad=True)
                else:
                    self.optimizer.second_step(zero_grad=True)
            return losses_first, pred_first
        else:

            predictions = self.model(data_dict)
            if type(self.model) is DDP:
                losses = self.model.module.get_losses(data_dict, predictions)
            else:
                losses = self.model.get_losses(data_dict, predictions)
            # self.optimizer.zero_grad()
            # losses['overall'].backward()
            # #self.model.module.set_mask_grad()
            # self.optimizer.step()
            if isinstance(self.optimizer, SAM):
                losses['overall'].backward()
                self.optimizer.first_step(zero_grad=True)
                losses2 = self.model.get_losses(data_dict, self.model(data_dict))
                losses2['overall'].backward()
                self.optimizer.second_step(zero_grad=True)
            elif isinstance(self.optimizer, PCGrad):
                self.optimizer.zero_grad()
                self.optimizer.pc_backward([losses['real_loss'], losses['fake_loss']])
                self.optimizer.step()
            else:
                self.optimizer.zero_grad()
                losses['overall'].backward()
                self.optimizer.step()


            return losses,predictions


    def train_epoch(
        self,
        epoch,
        train_data_loader,
        test_data_loaders=None,
        ):

        self.logger.info("===> Epoch[{}] start!".format(epoch))
        if epoch>=1:
            times_per_epoch = 2
        else:
            times_per_epoch = 2


        #times_per_epoch=4

        test_step = len(train_data_loader) // times_per_epoch    # test 10 times per epoch
        step_cnt = epoch * len(train_data_loader)

        # save the training data_dict
        data_dict = train_data_loader.dataset.data_dict
        self.save_data_dict('train', data_dict, ','.join(self.config['train_dataset']))
        # define training recorder
        train_recorder_loss = defaultdict(Recorder)
        train_recorder_metric = defaultdict(Recorder)

        for iteration, data_dict in tqdm(enumerate(train_data_loader),total=len(train_data_loader)):
            self.setTrain()
            # more elegant and more scalable way of moving data to GPU
            for key in data_dict.keys():
                if data_dict[key]!=None and key!='name':
                    data_dict[key]=data_dict[key].cuda()

            # ── Asymmetric Mixup (training only) ──────────────────────────
            if self.config.get('use_mixup', False):
                alpha   = self.config.get('mixup_alpha', 1.0)
                gamma   = self.config.get('mixup_gamma', 5.0)
                mixup_mode = self.config.get('mixup_mode', 'asymmetric')
                if mixup_mode == 'original':
                    data_dict['image'], data_dict['label_soft'] = asymmetric_mixup(
                        data_dict['image'], data_dict['label'], alpha=alpha, gamma=gamma,
                    )
                else:
                    mixup_k = self.config.get('mixup_k', 1)
                    data_dict = hardest_k_mixup(
                        self.model, data_dict, K=mixup_k, alpha=alpha, gamma=gamma,
                        selection=self.config.get('mixup_selection', 'hardest'),
                    )
            # ──────────────────────────────────────────────────────────────
            losses,predictions=self.train_step(data_dict)

            # update learning rate

            if 'SWA' in self.config and self.config['SWA'] and epoch>self.config['swa_start']:
                self.swa_model.update_parameters(self.model)

            # compute training metric for each batch data
            if type(self.model) is DDP:
                batch_metrics = self.model.module.get_train_metrics(data_dict, predictions)
            else:
                batch_metrics = self.model.get_train_metrics(data_dict, predictions)

            # store data by recorder
            ## store metric
            for name, value in batch_metrics.items():
                train_recorder_metric[name].update(value)
            ## store loss
            for name, value in losses.items():
                train_recorder_loss[name].update(value)

            # run tensorboard to visualize the training process
            if iteration % 300 == 0 and self.config['local_rank']==0:
                if self.config['SWA'] and (epoch>self.config['swa_start'] or self.config['dry_run']):
                    self.scheduler.step()
                # info for loss
                loss_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_loss.items():
                    v_avg = v.average()
                    if v_avg == None:
                        loss_str += f"training-loss, {k}: not calculated"
                        continue
                    loss_str += f"training-loss, {k}: {v_avg}    "
                    # tensorboard-1. loss
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_loss/{k}', v_avg, global_step=step_cnt)
                self.logger.info(loss_str)
                # info for metric
                metric_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_metric.items():
                    v_avg = v.average()
                    if v_avg == None:
                        metric_str += f"training-metric, {k}: not calculated    "
                        continue
                    metric_str += f"training-metric, {k}: {v_avg}    "
                    # tensorboard-2. metric
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_metric/{k}', v_avg, global_step=step_cnt)
                self.logger.info(metric_str)



                # clear recorder.
                # Note we only consider the current 300 samples for computing batch-level loss/metric
                for name, recorder in train_recorder_loss.items():  # clear loss recorder
                    recorder.clear()
                for name, recorder in train_recorder_metric.items():  # clear metric recorder
                    recorder.clear()

            # run test
            #if True:
            if (step_cnt+1) % test_step == 0:
                if test_data_loaders is not None and (not self.config['ddp'] ):
                    self.logger.info("===> Test start!")
                    test_best_metric = self.test_epoch(
                        epoch,
                        iteration,
                        test_data_loaders,
                        step_cnt,
                    )
                elif test_data_loaders is not None and (self.config['ddp'] and dist.get_rank() == 0):
                    self.logger.info("===> Test start!")
                    test_best_metric = self.test_epoch(
                        epoch,
                        iteration,
                        test_data_loaders,
                        step_cnt,
                    )
                else:
                    test_best_metric = None

                    # total_end_time = time.time()
            # total_elapsed_time = total_end_time - total_start_time
            # print("总花费的时间: {:.2f} 秒".format(total_elapsed_time))
            step_cnt += 1
        return test_best_metric

    def get_respect_acc(self,prob,label):
        pred = np.where(prob > 0.5, 1, 0)
        judge = (pred == label)
        zero_num = len(label) - np.count_nonzero(label)
        acc_fake = np.count_nonzero(judge[zero_num:]) / len(judge[zero_num:])
        acc_real = np.count_nonzero(judge[:zero_num]) / len(judge[:zero_num])
        return acc_real,acc_fake

    def test_one_dataset(self, data_loader):
        # define test recorder
        test_recorder_loss = defaultdict(Recorder)
        prediction_lists = []
        feature_lists=[]
        label_lists = []
        for i, data_dict in tqdm(enumerate(data_loader),total=len(data_loader)):
            # get data
            if 'label_spe' in data_dict:
                data_dict.pop('label_spe')  # remove the specific label
            data_dict['label'] = torch.where(data_dict['label']!=0, 1, 0)  # fix the label to 0 and 1 only
            # move data to GPU elegantly
            for key in data_dict.keys():
                if data_dict[key]!=None:
                    data_dict[key]=data_dict[key].cuda()
            # model forward without considering gradient computation
            predictions = self.inference(data_dict)
            label_lists += list(data_dict['label'].cpu().detach().numpy())
            prediction_lists += list(predictions['prob'].cpu().detach().numpy())
            feature_lists += list(predictions['feat'].cpu().detach().numpy())
            if type(self.model) is not AveragedModel:
                # compute all losses for each batch data
                with torch.no_grad():
                    if type(self.model) is DDP:
                        losses = self.model.module.get_losses(data_dict, predictions)
                    else:
                        losses = self.model.get_losses(data_dict, predictions)

                # store data by recorder
                for name, value in losses.items():
                    test_recorder_loss[name].update(value)

        return test_recorder_loss, np.array(prediction_lists), np.array(label_lists),np.array(feature_lists)

    def save_best(self,epoch,iteration,step,losses_one_dataset_recorder,key,metric_one_dataset):
        best_metric = self.best_metrics_all_time[key].get(self.metric_scoring,
                                                          float('-inf') if self.metric_scoring != 'eer' else float(
                                                              'inf'))
        # Check if the current score is an improvement
        improved = (metric_one_dataset[self.metric_scoring] > best_metric) if self.metric_scoring != 'eer' else (
                    metric_one_dataset[self.metric_scoring] < best_metric)
        if improved:
            # Update the best metric
            self.best_metrics_all_time[key][self.metric_scoring] = metric_one_dataset[self.metric_scoring]
            if key == 'avg':
                self.best_metrics_all_time[key]['dataset_dict'] = metric_one_dataset['dataset_dict']
            # Save checkpoint, feature, and metrics if specified in config
            if self.config['save_ckpt'] and key not in FFpp_pool:
                self.save_ckpt('test', key, f"{epoch}+{iteration}")
            self.save_metrics('test', metric_one_dataset, key)
        if losses_one_dataset_recorder is not None:
            # info for each dataset
            loss_str = f"dataset: {key}    step: {step}    "
            for k, v in losses_one_dataset_recorder.items():
                writer = self.get_writer('test', key, k)
                v_avg = v.average()
                if v_avg == None:
                    print(f'{k} is not calculated')
                    continue
                # tensorboard-1. loss
                writer.add_scalar(f'test_losses/{k}', v_avg, global_step=step)
                loss_str += f"testing-loss, {k}: {v_avg}    "
            self.logger.info(loss_str)
        # tqdm.write(loss_str)
        metric_str = f"dataset: {key}    step: {step}    "
        for k, v in metric_one_dataset.items():
            if k == 'pred' or k == 'label' or k=='dataset_dict':
                continue
            metric_str += f"testing-metric, {k}: {v}    "
            # tensorboard-2. metric
            writer = self.get_writer('test', key, k)
            writer.add_scalar(f'test_metrics/{k}', v, global_step=step)
        if 'pred' in metric_one_dataset:
            acc_real, acc_fake = self.get_respect_acc(metric_one_dataset['pred'], metric_one_dataset['label'])
            metric_str += f'testing-metric, acc_real:{acc_real}; acc_fake:{acc_fake}'
            writer.add_scalar(f'test_metrics/acc_real', acc_real, global_step=step)
            writer.add_scalar(f'test_metrics/acc_fake', acc_fake, global_step=step)
        self.logger.info(metric_str)

    def test_epoch(self, epoch, iteration, test_data_loaders, step):
        # set model to eval mode
        self.setEval()

        # define test recorder
        losses_all_datasets = {}
        metrics_all_datasets = {}
        best_metrics_per_dataset = defaultdict(dict)  # best metric for each dataset, for each metric
        avg_metric = {'acc': 0, 'auc': 0, 'eer': 0, 'ap': 0,'video_auc': 0,'dataset_dict':{}}
        # testing for all test data
        keys = test_data_loaders.keys()
        for key in keys:
            # save the testing data_dict
            data_dict = test_data_loaders[key].dataset.data_dict
            self.save_data_dict('test', data_dict, key)

            # compute loss for each dataset
            losses_one_dataset_recorder, predictions_nps, label_nps, feature_nps = self.test_one_dataset(test_data_loaders[key])
            # print(f'stack len:{predictions_nps.shape};{label_nps.shape};{len(data_dict["image"])}')
            losses_all_datasets[key] = losses_one_dataset_recorder
            metric_one_dataset=get_test_metrics(y_pred=predictions_nps,y_true=label_nps,img_names=data_dict['image'])

            # Adaptive threshold (OWTTT) for zero-shot NTTA
            model_module = self.model.module if hasattr(self.model, 'module') else self.model
            if self.config.get('use_adaptive_threshold', True) and hasattr(model_module, 'compute_adaptive_threshold'):
                model_module.prediction_queue.extend(predictions_nps.tolist())
                if len(model_module.prediction_queue) > 1000:
                    model_module.prediction_queue = model_module.prediction_queue[-1000:]
                adaptive_th = model_module.compute_adaptive_threshold()
                metric_one_dataset['acc_adaptive'] = float(
                    np.mean((predictions_nps > adaptive_th).astype(int) == label_nps)
                )

            for metric_name, value in metric_one_dataset.items():
                if metric_name in avg_metric:
                    avg_metric[metric_name]+=value
            avg_metric['dataset_dict'][key] = metric_one_dataset[self.metric_scoring]
            if type(self.model) is AveragedModel:
                metric_str = f"Iter Final for SWA:    "
                for k, v in metric_one_dataset.items():
                    metric_str += f"testing-metric, {k}: {v}    "
                self.logger.info(metric_str)
                continue
            self.save_best(epoch,iteration,step,losses_one_dataset_recorder,key,metric_one_dataset)

        if len(keys)>0 and self.config.get('save_avg',False):
            # calculate avg value
            for key in avg_metric:
                if key != 'dataset_dict':
                    avg_metric[key] /= len(keys)
            self.save_best(epoch, iteration, step, None, 'avg', avg_metric)

        self.logger.info('===> Test Done!')
        return self.best_metrics_all_time  # return all types of mean metrics for determining the best ckpt

    @torch.no_grad()
    def inference(self, data_dict):
        predictions = self.model(data_dict, inference=True)
        return predictions