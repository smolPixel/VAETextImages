# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : seqgan_instructor.py
# @Time         : Created at 2019-06-05
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

# from modelsGAN.SeqGAN_D import SeqGAN_D
# from modelsGAN.SeqGAN_G import SeqGAN_G
# from utils import rollout
# from utils.data_loader import GenDataIter, DisDataIter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# import config as cfg
from Generator.SeqGAN.utils import Signal
from Decoders.LSTMGAN import SeqGAN_G
from torch.utils.data import DataLoader
from Discriminators.CNNDiscriminatorGAN import SeqGAN_D
from multiprocessing import cpu_count
# from metrics.bleu import BLEU
# from metrics.clas_acc import ACC
# from metrics.nll import NLL
# from metrics.ppl import PPL
# from utils.cat_data_loader import CatClasDataIter
# from utils.data_loader import GenDataIter
# from utils.helpers import Signal, create_logger, get_fixed_temperature
# from utils.text_process import load_dict, write_tokens, tensor_to_tokens


class SeqGANInstructor:
    def __init__(self, argdict, datasets):
        self.sig = Signal()
        self.argdict = argdict

        self.clas = None
        self.datasets=datasets
        self.training_set=self.datasets['train']
        self.dev_set=self.datasets['dev']
        self.test_set=self.datasets['test']
        #
        # # Dataloader
        # try:
        #     self.train_data = GenDataIter(cfg.train_data)
        #     self.test_data = GenDataIter(cfg.test_data, if_test_data=True)
        # except:
        #     pass
        #
        # try:
        #     self.train_data_list = [GenDataIter(cfg.cat_train_data.format(i)) for i in range(cfg.k_label)]
        #     self.test_data_list = [GenDataIter(cfg.cat_test_data.format(i), if_test_data=True) for i in
        #                            range(cfg.k_label)]
        #     self.clas_data_list = [GenDataIter(cfg.cat_test_data.format(str(i)), if_test_data=True) for i in
        #                            range(cfg.k_label)]
        #
        #     self.train_samples_list = [self.train_data_list[i].target for i in range(cfg.k_label)]
        #     self.clas_samples_list = [self.clas_data_list[i].target for i in range(cfg.k_label)]
        # except:
        #     pass

        # Criterion
        self.mle_criterion = nn.NLLLoss()
        self.dis_criterion = nn.CrossEntropyLoss()
        self.clas_criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.clas_opt = None

        # Metrics
        # self.bleu = BLEU('BLEU', gram=[2, 3, 4, 5], if_use=cfg.use_bleu)
        # self.nll_gen = NLL('NLL_gen', if_use=cfg.use_nll_gen, gpu=cfg.CUDA)
        # self.nll_div = NLL('NLL_div', if_use=cfg.use_nll_div, gpu=cfg.CUDA)
        # self.self_bleu = BLEU('Self-BLEU', gram=[2, 3, 4], if_use=cfg.use_self_bleu)
        # self.clas_acc = ACC(if_use=cfg.use_clas_acc)
        # self.ppl = PPL(self.train_data, self.test_data, n_gram=5, if_use=cfg.use_ppl)
        # self.all_metrics = [self.bleu, self.nll_gen, self.nll_div, self.self_bleu, self.ppl]
        self.vocab_size=self.training_set.vocab_size
        self.gen = SeqGAN_G(argdict, 300, 1024, self.vocab_size, 32, self.training_set.pad_idx)
        self.dis = SeqGAN_D(argdict, 300, self.vocab_size, self.training_set.pad_idx)
        self.init_model()

        # Optimizer
        self.gen_opt = optim.Adam(self.gen.parameters(), 0.01)
        self.gen_adv_opt = optim.Adam(self.gen.parameters(), 0.01)
        self.dis_opt = optim.Adam(self.dis.parameters(), 1e-4)


    def init_model(self):
        # if cfg.dis_pretrain:
        #     self.log.info('Load pre-trained discriminator: {}'.format(cfg.pretrained_dis_path))
        #     self.dis.load_state_dict(torch.load(cfg.pretrained_dis_path, map_location='cuda:{}'.format(cfg.device)))
        # if cfg.gen_pretrain:
        #     self.log.info('Load MLE pre-trained generator: {}'.format(cfg.pretrained_gen_path))
        #     self.gen.load_state_dict(torch.load(cfg.pretrained_gen_path, map_location='cuda:{}'.format(cfg.device)))
        #
        # if cfg.CUDA:
        self.gen = self.gen.cuda()
        self.dis = self.dis.cuda()

    def train_gen_epoch(self, model, dataset, criterion, optimizer):
        total_loss = 0

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.argdict['batch_size'],  # self.argdict.batch_size,
            shuffle=True,
            num_workers=cpu_count(),
            pin_memory=torch.cuda.is_available()
        )

        for i, data in enumerate(data_loader):
            inp, target = torch.Tensor(data['input']).type(torch.LongTensor), torch.Tensor(data['target']).type(torch.LongTensor)
            inp, target = inp.cuda(), target.cuda()
            bs=inp.shape[0]

            hidden = model.init_hidden(bs)
            pred = model.forward(inp, hidden)
            loss = criterion(pred, target.view(-1))
            self.optimize(optimizer, loss, model)
            total_loss += loss.item()
        return total_loss / len(data_loader)

    def train_dis_epoch(self, model, data_loader, criterion, optimizer):
        total_loss = 0
        total_acc = 0
        total_num = 0
        for i, data in enumerate(data_loader):
            inp, target = data['input'], data['target']
            if cfg.CUDA:
                inp, target = inp.cuda(), target.cuda()

            pred = model.forward(inp)
            loss = criterion(pred, target)
            self.optimize(optimizer, loss, model)

            total_loss += loss.item()
            total_loss += loss.item()
            total_acc += torch.sum((pred.argmax(dim=-1) == target)).item()
            total_num += inp.size(0)

        total_loss /= len(data_loader)
        total_acc /= total_num
        return total_loss, total_acc

    def train_classifier(self, epochs):
        """
        Classifier for calculating the classification accuracy metric of category text generation.

        Note: the train and test data for the classifier is opposite to the generator.
        Because the classifier is to calculate the classification accuracy of the generated samples
        where are trained on self.train_samples_list.

        Since there's no test data in synthetic data (oracle data), the synthetic data experiments
        doesn't need a classifier.
        """
        import copy

        # Prepare data for Classifier
        clas_data = CatClasDataIter(self.clas_samples_list)
        eval_clas_data = CatClasDataIter(self.train_samples_list)

        max_acc = 0
        best_clas = None
        for epoch in range(epochs):
            c_loss, c_acc = self.train_dis_epoch(self.clas, clas_data.loader, self.clas_criterion,
                                                 self.clas_opt)
            _, eval_acc = self.eval_dis(self.clas, eval_clas_data.loader, self.clas_criterion)
            if eval_acc > max_acc:
                best_clas = copy.deepcopy(self.clas.state_dict())  # save the best classifier
                max_acc = eval_acc
            self.log.info('[PRE-CLAS] epoch %d: c_loss = %.4f, c_acc = %.4f, eval_acc = %.4f, max_eval_acc = %.4f',
                          epoch, c_loss, c_acc, eval_acc, max_acc)
        self.clas.load_state_dict(copy.deepcopy(best_clas))  # Reload the best classifier

    @staticmethod
    def eval_dis(model, data_loader, criterion):
        total_loss = 0
        total_acc = 0
        total_num = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inp, target = data['input'], data['target']
                if cfg.CUDA:
                    inp, target = inp.cuda(), target.cuda()

                pred = model.forward(inp)
                loss = criterion(pred, target)
                total_loss += loss.item()
                total_acc += torch.sum((pred.argmax(dim=-1) == target)).item()
                total_num += inp.size(0)
            total_loss /= len(data_loader)
            total_acc /= total_num
        return total_loss, total_acc

    @staticmethod
    def optimize_multi(opts, losses):
        for i, (opt, loss) in enumerate(zip(opts, losses)):
            opt.zero_grad()
            loss.backward(retain_graph=True if i < len(opts) - 1 else False)
            opt.step()

    @staticmethod
    def optimize(opt, loss, model=None, retain_graph=False):
        opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), model.argdict['clip_norm'])
        opt.step()


    def cal_metrics(self, fmt_str=False):
        """
        Calculate metrics
        :param fmt_str: if return format string for logging
        """
        with torch.no_grad():
            # Prepare data for evaluation
            eval_samples = self.gen.sample(self.argdict['samples_num'], 4 * self.argdict['batch_size'])
            gen_data = GenDataIter(eval_samples)
            gen_tokens = tensor_to_tokens(eval_samples, self.idx2word_dict)
            gen_tokens_s = tensor_to_tokens(self.gen.sample(200, 200), self.idx2word_dict)

            # Reset metrics
            self.bleu.reset(test_text=gen_tokens, real_text=self.test_data.tokens)
            self.nll_gen.reset(self.gen, self.train_data.loader)
            self.nll_div.reset(self.gen, gen_data.loader)
            self.self_bleu.reset(test_text=gen_tokens_s, real_text=gen_tokens)
            self.ppl.reset(gen_tokens)

        if fmt_str:
            return ', '.join(['%s = %s' % (metric.get_name(), metric.get_score()) for metric in self.all_metrics])
        else:
            return [metric.get_score() for metric in self.all_metrics]

    def cal_metrics_with_label(self, label_i):
        assert type(label_i) == int, 'missing label'

        with torch.no_grad():
            # Prepare data for evaluation
            eval_samples = self.gen.sample(cfg.samples_num, 8 * cfg.batch_size, label_i=label_i)
            gen_data = GenDataIter(eval_samples)
            gen_tokens = tensor_to_tokens(eval_samples, self.idx2word_dict)
            gen_tokens_s = tensor_to_tokens(self.gen.sample(200, 200, label_i=label_i), self.idx2word_dict)
            clas_data = CatClasDataIter([eval_samples], label_i)

            # Reset metrics
            self.bleu.reset(test_text=gen_tokens, real_text=self.test_data_list[label_i].tokens)
            self.nll_gen.reset(self.gen, self.train_data_list[label_i].loader, label_i)
            self.nll_div.reset(self.gen, gen_data.loader, label_i)
            self.self_bleu.reset(test_text=gen_tokens_s, real_text=gen_tokens)
            self.clas_acc.reset(self.clas, clas_data.loader)
            self.ppl.reset(gen_tokens)

        return [metric.get_score() for metric in self.all_metrics]

    def comb_metrics(self, fmt_str=False):
        all_scores = [self.cal_metrics_with_label(label_i) for label_i in range(cfg.k_label)]
        all_scores = np.array(all_scores).T.tolist()  # each row for each metric

        if fmt_str:
            return ', '.join(['%s = %s' % (metric.get_name(), score)
                              for (metric, score) in zip(self.all_metrics, all_scores)])
        return all_scores

    def _save(self, phase, epoch):
        """Save model state dict and generator's samples"""
        if phase != 'ADV':
            torch.save(self.gen.state_dict(), cfg.save_model_root + 'gen_{}_{:05d}.pt'.format(phase, epoch))
        save_sample_path = cfg.save_samples_root + 'samples_{}_{:05d}.txt'.format(phase, epoch)
        samples = self.gen.sample(cfg.batch_size, cfg.batch_size)
        write_tokens(save_sample_path, tensor_to_tokens(samples, self.idx2word_dict))

    def update_temperature(self, i, N):
        self.gen.temperature.data = torch.Tensor([get_fixed_temperature(cfg.temperature, i, N, cfg.temp_adpt)])
        if cfg.CUDA:
            self.gen.temperature.data = self.gen.temperature.data.cuda()

    def _run(self):
        # ===PRE-TRAINING===
        # TRAIN GENERATOR
        if not self.argdict['gen_pretrain']:
            print('Starting Generator MLE Training...')
            self.pretrain_generator(self.argdict['MLE_train_epoch'])
            # if cfg.if_save and not cfg.if_test:
            #     torch.save(self.gen.state_dict(), cfg.pretrained_gen_path)
            #     print('Save pre-trained generator: {}'.format(cfg.pretrained_gen_path))

        # ===TRAIN DISCRIMINATOR====
        if not cfg.dis_pretrain:
            self.log.info('Starting Discriminator Training...')
            self.train_discriminator(cfg.d_step, cfg.d_epoch)
            if cfg.if_save and not cfg.if_test:
                torch.save(self.dis.state_dict(), cfg.pretrained_dis_path)
                print('Save pre-trained discriminator: {}'.format(cfg.pretrained_dis_path))

        # ===ADVERSARIAL TRAINING===
        self.log.info('Starting Adversarial Training...')
        self.log.info('Initial generator: %s' % (self.cal_metrics(fmt_str=True)))

        for adv_epoch in range(cfg.ADV_train_epoch):
            self.log.info('-----\nADV EPOCH %d\n-----' % adv_epoch)
            self.sig.update()
            if self.sig.adv_sig:
                self.adv_train_generator(cfg.ADV_g_step)  # Generator
                self.train_discriminator(cfg.ADV_d_step, cfg.ADV_d_epoch, 'ADV')  # Discriminator

                if adv_epoch % cfg.adv_log_step == 0 or adv_epoch == cfg.ADV_train_epoch - 1:
                    if cfg.if_save and not cfg.if_test:
                        self._save('ADV', adv_epoch)
            else:
                self.log.info('>>> Stop by adv_signal! Finishing adversarial training...')
                break
        self._sample()

    def _test(self):
        print('>>> Begin test...')

        self._run()
        pass

    def pretrain_generator(self, epochs):
        """
        Max Likelihood Pre-training for the generator
        """
        for epoch in range(epochs):
            self.sig.update()
            if self.sig.pre_sig:
                pre_loss = self.train_gen_epoch(self.gen, self.datasets['train'], self.mle_criterion, self.gen_opt)
                print('[MLE-GEN] epoch %d : pre_loss = %.4f, %s' % (epoch, pre_loss, self.cal_metrics(fmt_str=True)))
                # ===Test===
                # if epoch % cfg.pre_log_step == 0 or epoch == epochs - 1:
                #     self.log.info(
                #         '[MLE-GEN] epoch %d : pre_loss = %.4f, %s' % (epoch, pre_loss, self.cal_metrics(fmt_str=True)))
                #     if cfg.if_save and not cfg.if_test:
                #         self._save('MLE', epoch)
            else:
                print('>>> Stop by pre signal, skip to adversarial training...')
                break

    def adv_train_generator(self, g_step):
        """
        The gen is trained using policy gradients, using the reward from the discriminator.
        Training is done for num_batches batches.
        """
        rollout_func = rollout.ROLLOUT(self.gen, cfg.CUDA)
        total_g_loss = 0
        for step in range(g_step):
            inp, target = GenDataIter.prepare(self.gen.sample(cfg.batch_size, cfg.batch_size), gpu=cfg.CUDA)

            # ===Train===
            rewards = rollout_func.get_reward(target, cfg.rollout_num, self.dis)
            adv_loss = self.gen.batchPGLoss(inp, target, rewards)
            self.optimize(self.gen_adv_opt, adv_loss)
            total_g_loss += adv_loss.item()

        # ===Test===
        self.log.info('[ADV-GEN]: g_loss = %.4f, %s' % (total_g_loss, self.cal_metrics(fmt_str=True)))

    def train_discriminator(self, d_step, d_epoch, phase='MLE'):
        """
        Training the discriminator on real_data_samples (positive) and generated samples from gen (negative).
        Samples are drawn d_step times, and the discriminator is trained for d_epoch d_epoch.
        """
        # prepare loader for validate
        global d_loss, train_acc
        for step in range(d_step):
            # prepare loader for training
            pos_samples = self.train_data.target
            neg_samples = self.gen.sample(cfg.samples_num, 4 * cfg.batch_size)
            dis_data = DisDataIter(pos_samples, neg_samples)

            for epoch in range(d_epoch):
                # ===Train===
                d_loss, train_acc = self.train_dis_epoch(self.dis, dis_data.loader, self.dis_criterion,
                                                         self.dis_opt)

            # ===Test===
            self.log.info('[%s-DIS] d_step %d: d_loss = %.4f, train_acc = %.4f,' % (
                phase, step, d_loss, train_acc))

            if cfg.if_save and not cfg.if_test:
                torch.save(self.dis.state_dict(), cfg.pretrained_dis_path)
#
# class SeqGANInstructor(BasicInstructor):
#     def __init__(self, opt):
#         super(SeqGANInstructor, self).__init__(opt)
#
#         # generator, discriminator
#         self.gen = SeqGAN_G(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
#                             cfg.padding_idx, gpu=cfg.CUDA)
#         self.dis = SeqGAN_D(cfg.dis_embed_dim, cfg.vocab_size, cfg.padding_idx, gpu=cfg.CUDA)
#         self.init_model()
#
#         # Optimizer
#         self.gen_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
#         self.gen_adv_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
#         self.dis_opt = optim.Adam(self.dis.parameters(), lr=cfg.dis_lr)
#
#     def _run(self):
#         # ===PRE-TRAINING===
#         # TRAIN GENERATOR
#         if not cfg.gen_pretrain:
#             self.log.info('Starting Generator MLE Training...')
#             self.pretrain_generator(cfg.MLE_train_epoch)
#             if cfg.if_save and not cfg.if_test:
#                 torch.save(self.gen.state_dict(), cfg.pretrained_gen_path)
#                 print('Save pre-trained generator: {}'.format(cfg.pretrained_gen_path))
#
#         # ===TRAIN DISCRIMINATOR====
#         if not cfg.dis_pretrain:
#             self.log.info('Starting Discriminator Training...')
#             self.train_discriminator(cfg.d_step, cfg.d_epoch)
#             if cfg.if_save and not cfg.if_test:
#                 torch.save(self.dis.state_dict(), cfg.pretrained_dis_path)
#                 print('Save pre-trained discriminator: {}'.format(cfg.pretrained_dis_path))
#
#         # ===ADVERSARIAL TRAINING===
#         self.log.info('Starting Adversarial Training...')
#         self.log.info('Initial generator: %s' % (self.cal_metrics(fmt_str=True)))
#
#         for adv_epoch in range(cfg.ADV_train_epoch):
#             self.log.info('-----\nADV EPOCH %d\n-----' % adv_epoch)
#             self.sig.update()
#             if self.sig.adv_sig:
#                 self.adv_train_generator(cfg.ADV_g_step)  # Generator
#                 self.train_discriminator(cfg.ADV_d_step, cfg.ADV_d_epoch, 'ADV')  # Discriminator
#
#                 if adv_epoch % cfg.adv_log_step == 0 or adv_epoch == cfg.ADV_train_epoch - 1:
#                     if cfg.if_save and not cfg.if_test:
#                         self._save('ADV', adv_epoch)
#             else:
#                 self.log.info('>>> Stop by adv_signal! Finishing adversarial training...')
#                 break
#         self._sample()
#
#     def _test(self):
#         print('>>> Begin test...')
#
#         self._run()
#         pass
#
#     def pretrain_generator(self, epochs):
#         """
#         Max Likelihood Pre-training for the generator
#         """
#         for epoch in range(epochs):
#             self.sig.update()
#             if self.sig.pre_sig:
#                 pre_loss = self.train_gen_epoch(self.gen, self.train_data.loader, self.mle_criterion, self.gen_opt)
#
#                 # ===Test===
#                 if epoch % cfg.pre_log_step == 0 or epoch == epochs - 1:
#                     self.log.info(
#                         '[MLE-GEN] epoch %d : pre_loss = %.4f, %s' % (epoch, pre_loss, self.cal_metrics(fmt_str=True)))
#                     if cfg.if_save and not cfg.if_test:
#                         self._save('MLE', epoch)
#             else:
#                 self.log.info('>>> Stop by pre signal, skip to adversarial training...')
#                 break
#
#     def adv_train_generator(self, g_step):
#         """
#         The gen is trained using policy gradients, using the reward from the discriminator.
#         Training is done for num_batches batches.
#         """
#         rollout_func = rollout.ROLLOUT(self.gen, cfg.CUDA)
#         total_g_loss = 0
#         for step in range(g_step):
#             inp, target = GenDataIter.prepare(self.gen.sample(cfg.batch_size, cfg.batch_size), gpu=cfg.CUDA)
#
#             # ===Train===
#             rewards = rollout_func.get_reward(target, cfg.rollout_num, self.dis)
#             adv_loss = self.gen.batchPGLoss(inp, target, rewards)
#             self.optimize(self.gen_adv_opt, adv_loss)
#             total_g_loss += adv_loss.item()
#
#         # ===Test===
#         self.log.info('[ADV-GEN]: g_loss = %.4f, %s' % (total_g_loss, self.cal_metrics(fmt_str=True)))
#
#     def train_discriminator(self, d_step, d_epoch, phase='MLE'):
#         """
#         Training the discriminator on real_data_samples (positive) and generated samples from gen (negative).
#         Samples are drawn d_step times, and the discriminator is trained for d_epoch d_epoch.
#         """
#         # prepare loader for validate
#         global d_loss, train_acc
#         for step in range(d_step):
#             # prepare loader for training
#             pos_samples = self.train_data.target
#             neg_samples = self.gen.sample(cfg.samples_num, 4 * cfg.batch_size)
#             dis_data = DisDataIter(pos_samples, neg_samples)
#
#             for epoch in range(d_epoch):
#                 # ===Train===
#                 d_loss, train_acc = self.train_dis_epoch(self.dis, dis_data.loader, self.dis_criterion,
#                                                          self.dis_opt)
#
#             # ===Test===
#             self.log.info('[%s-DIS] d_step %d: d_loss = %.4f, train_acc = %.4f,' % (
#                 phase, step, d_loss, train_acc))
#
#             if cfg.if_save and not cfg.if_test:
#                 torch.save(self.dis.state_dict(), cfg.pretrained_dis_path)
