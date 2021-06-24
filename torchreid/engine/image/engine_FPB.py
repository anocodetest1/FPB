from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import datetime

import torch

import torchreid
from torchreid.engine import engine
from torchreid.losses import CrossEntropyLoss, TripletLoss, CenterLoss
from torchreid.utils import AverageMeter, open_specified_layers, open_all_layers
from torchreid import metrics


class ImageFPBEngine(engine.Engine):
    r"""PcbAttentionEngine
    """   

    def __init__(self, datamanager, model, optimizer, margin=0.3,
                 weight_t=1, weight_x=1, scheduler=None, use_gpu=True,
                 label_smooth=True, div_penalty=None, div_start=0):
        super(ImageFPBEngine, self).__init__(datamanager, model, optimizer, scheduler, use_gpu)

        self.weight_t = weight_t
        self.weight_x = weight_x
        self.num_parts = 3
        self.feature_dim = 2048+(1024)*self.num_parts
        self.centloss_weight = 1.0+self.num_parts*(1.0)

        self.div_penalty = div_penalty
        self.div_start = div_start

        self.criterion_t = TripletLoss(margin=margin)
        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )
        self.criterion_c = CenterLoss(num_classes=751, feat_dim=self.feature_dim) 


    def train(self, epoch, max_epoch, trainloader, fixbase_epoch=0, open_layers=None, print_freq=10):
        losses_t = AverageMeter()
        losses_x = AverageMeter()
        losses_c = AverageMeter()
        losses_p = AverageMeter()
        accs = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()


        if self.div_penalty is not None and epoch >= self.div_start:
            print("Using div penalty!")

        self.model.train()
        if (epoch+1)<=fixbase_epoch and open_layers is not None:
            print('* Only train {} (epoch: {}/{})'.format(open_layers, epoch+1, fixbase_epoch))
            open_specified_layers(self.model, open_layers)
        else:
            open_all_layers(self.model)

        num_batches = len(trainloader)
        end = time.time()
        for batch_idx, data in enumerate(trainloader):
            data_time.update(time.time() - end)

            imgs, pids = self._parse_data_for_train(data)
            if self.use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()
            
            self.optimizer.zero_grad()
            output, fea, reg_feat = self.model(imgs)
            
            b = output[0].size(0)   #
            loss_c = self._compute_loss(self.criterion_c, fea, pids)
            loss_t = self._compute_loss(self.criterion_t, fea, pids)
            loss_x = self._compute_loss(self.criterion_x, output, pids)
            loss = self.weight_x * loss_x + self.weight_t * loss_t + 0.0005 / self.centloss_weight * loss_c  

            if self.div_penalty is not None:
                penalty = self.div_penalty(reg_feat)

                if epoch >= self.div_start:
                    loss += penalty

                losses_p.update(penalty.item(), b)

            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)

            losses_t.update(loss_t.item(), b)
            losses_x.update(loss_x.item(), b)
            losses_c.update(loss_c.item(), b)
          
            accs.update(metrics.accuracy(output, pids)[0].item())

            if (batch_idx+1) % print_freq == 0:
                # estimate remaining time
                eta_seconds = batch_time.avg * (num_batches-(batch_idx+1) + (max_epoch-(epoch+1))*num_batches)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss_t {loss_t.val:.4f} ({loss_t.avg:.4f})\t'
                      'Loss_x {loss_x.val:.4f} ({loss_x.avg:.4f})\t'
                      'Loss_c {loss_c.val:.4f} ({loss_c.avg:.4f})\t'
                      'Acc {acc.val:.2f} ({acc.avg:.2f})\t'
                      'Lr {lr:.6f}\t'
                      'eta {eta}'.format(
                      epoch+1, max_epoch, batch_idx+1, num_batches,
                      batch_time=batch_time,
                      data_time=data_time,
                      loss_t=losses_t,
                      loss_x=losses_x,
                      loss_c=losses_c,
                      acc=accs,
                      lr=self.optimizer.param_groups[0]['lr'],
                      eta=eta_str
                    )
                )

            if self.writer is not None:
                n_iter = epoch * num_batches + batch_idx
                self.writer.add_scalar('Train/Time', batch_time.avg, n_iter)
                self.writer.add_scalar('Train/Data', data_time.avg, n_iter)
                self.writer.add_scalar('Train/Loss_t', losses_t.val, n_iter)
                self.writer.add_scalar('Train/Loss_x', losses_x.val, n_iter)
                self.writer.add_scalar('Train/Loss_c', losses_c.val, n_iter)
                self.writer.add_scalar('Train/Acc1', accs.val, n_iter)
                self.writer.add_scalar('Train/Lr', self.optimizer.param_groups[0]['lr'], n_iter)
                if self.div_penalty is not None:        
                    self.writer.add_scalar('Train/Loss_p', losses_p.val, n_iter)
            
            end = time.time()

        if self.scheduler is not None:
            self.scheduler.step()
