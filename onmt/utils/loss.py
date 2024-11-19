
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

import onmt
import onmt.inputters as inputters


def build_loss_compute(model, tgt_vocab, opt, train=True):

    device = torch.device("cuda" if onmt.utils.misc.use_gpu(opt) else "cpu")
    compute = S2SLossCompute(model.generator, tgt_vocab)
    compute.to(device)

    return compute


class S2SLossCompute(nn.Module):

    def __init__(self, generator, tgt_vocab):
        super(S2SLossCompute, self).__init__()
        self.generator = generator
        self.tgt_vocab = tgt_vocab
        self.padding_idx = tgt_vocab.stoi[inputters.PAD_WORD]
        self.criterion = nn.NLLLoss(
            ignore_index=self.padding_idx, reduction='sum')

    def compute_loss(self, batch, output1, attns1, output2, attns2, output3, attns3, normalization, dis1_sim, dis2_sim, dis3_sim, dis12_sim, dis13_sim, dis23_sim):
        target = batch.tgt[0][1:]
        
        bottled_output1 = self._bottle(output1)
        scores1 = self.generator(bottled_output1)        
        bottled_output2 = self._bottle(output2)
        scores2 = self.generator(bottled_output2)        
        bottled_output3 = self._bottle(output3)
        scores3 = self.generator(bottled_output3)
        
        gtruth = target.view(-1)

        loss1 = self.criterion(scores1, gtruth) 
        loss2 = self.criterion(scores2, gtruth) 
        loss3 = self.criterion(scores3, gtruth) 
        loss = loss1 + loss2 + loss3 + 0.0001*((1 - dis1_sim) + (1- dis2_sim) + (1 - dis3_sim))#+ 0.0001*(dis12_sim + dis13_sim + dis23_sim)   
        loss = loss.div(float(3))
        loss.div(float(normalization)).backward()
  
        actual_loss = loss.data.clone().item()
        norm_loss = loss.data.clone().item()/ batch.batch_size
        stats = self._stats(actual_loss, norm_loss, scores1 + scores2 + scores3, gtruth)
        return stats

    def monolithic_compute_loss(self, batch, output1, attns1, output2, attns2, output3, attns3, dis1_sim, dis2_sim, dis3_sim, dis12_sim, dis13_sim, dis23_sim):

        target = batch.tgt[0][1:]
        
        bottled_output1 = self._bottle(output1)
        scores1 = self.generator(bottled_output1)
        bottled_output2 = self._bottle(output2)
        scores2 = self.generator(bottled_output2)
        bottled_output3 = self._bottle(output3)
        scores3 = self.generator(bottled_output3)        
        
        gtruth = target.view(-1)
        
        loss1 = self.criterion(scores1, gtruth) 
        loss2 = self.criterion(scores2, gtruth) 
        loss3 = self.criterion(scores3, gtruth)
        loss = loss1 + loss2 + loss3 + 0.0001*((1-dis1_sim) + (1-dis2_sim) + (1 -dis3_sim))#+ 0.001*(dis12_sim + dis13_sim + dis23_sim) #
        
        loss = loss.div(float(3))
        
        actual_loss = loss.clone().item()
        norm_loss = loss.clone().item()/ batch.batch_size
        stats = self._stats(actual_loss, norm_loss, scores1+scores2+scores3, gtruth)
        return stats

    def _stats(self, actual_loss, norm_loss, scores, target):

        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum() \
                          .item()
        num_non_padding = non_padding.sum().item()
        return onmt.utils.Statistics(actual_loss, norm_loss, num_non_padding, num_correct)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class LossComputeBase(nn.Module):


    def __init__(self, generator, tgt_vocab):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.tgt_vocab = tgt_vocab
        self.padding_idx = tgt_vocab.stoi[inputters.PAD_WORD]

    def _make_shard_state(self, batch, output, range_, attns=None):

        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):

        return NotImplementedError

    def monolithic_compute_loss(self, batch, output, attns):

        range_ = (0, batch.tgt.size(0))
        shard_state = self._make_shard_state(batch, output, range_, attns)
        _, batch_stats = self._compute_loss(batch, **shard_state)

        return batch_stats

    def sharded_compute_loss(self, batch, output, attns,
                             cur_trunc, trunc_size, shard_size,
                             normalization):

        batch_stats = onmt.utils.Statistics()
        range_ = (cur_trunc, cur_trunc + trunc_size)
        shard_state = self._make_shard_state(batch, output, range_, attns)
        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, **shard)
            loss.div(float(normalization)).backward()
            batch_stats.update(stats)
        return batch_stats

    def _stats(self, loss, scores, target):

        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum() \
                          .item()
        num_non_padding = non_padding.sum().item()
        return onmt.utils.Statistics(loss.item(), num_non_padding, num_correct)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class LabelSmoothingLoss(nn.Module):

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):

        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, generator, tgt_vocab, normalization="sents",
                 label_smoothing=0.0):
        super(NMTLossCompute, self).__init__(generator, tgt_vocab)
        self.sparse = not isinstance(generator[1], nn.LogSoftmax)
        if label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(
                label_smoothing, len(tgt_vocab), ignore_index=self.padding_idx
            )
        # elif self.sparse:
        #     self.criterion = SparsemaxLoss(
        #         ignore_index=self.padding_idx, size_average=False
        #     )
        else:
            self.criterion = nn.NLLLoss(
                ignore_index=self.padding_idx, reduction='sum'
            )

    def _make_shard_state(self, batch, output, range_, attns=None):
        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
        }

    def _compute_loss(self, batch, output, target):
        bottled_output = self._bottle(output)
        if self.sparse:
            # for sparsemax loss, the loss function operates on the raw output
            # vector, not a probability vector. Hence it's only necessary to
            # apply the first part of the generator here.
            scores = self.generator[0](bottled_output)
        else:
            scores = self.generator(bottled_output)
        gtruth = target.view(-1)

        loss = self.criterion(scores, gtruth)
        stats = self._stats(loss.clone(), scores, gtruth)

        return loss, stats


def filter_shard_state(state, shard_size=None):
    """ ? """
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):

    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
