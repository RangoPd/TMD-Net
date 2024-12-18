from __future__ import division, unicode_literals
from __future__ import print_function

import torch
import onmt.inputters as inputters
from eval.eval import eval

class TranslationBuilder(object):

    def __init__(self, data, data_type, fields, n_best=1, replace_unk=False,
                 has_tgt=False):
        self.data = data
        self.data_type = data_type
        self.fields = fields
        self.n_best = n_best
        self.replace_unk = replace_unk
        self.has_tgt = has_tgt

    def _build_target_tokens(self, src, src_vocab, src_raw, pred, attn):
        vocab = self.fields["tgt"].vocab
        tokens = []
        for tok in pred:
            if tok < len(vocab):
                tokens.append(vocab.itos[tok])
            else:
                tokens.append(src_vocab.itos[tok - len(vocab)])
            if tokens[-1] == inputters.EOS_WORD:
                tokens = tokens[:-1]
                break
        if self.replace_unk and (attn is not None) and (src is not None):
            for i in range(len(tokens)):
                if tokens[i] == vocab.itos[inputters.UNK]:
                    _, max_index = attn[i].max(0)
                    max_index_i = max_index.item() // src.size(1)
                    max_index_j = max_index.item() - max_index_i * src.size(1)
                    if max_index_i < len(src_raw):
                        if max_index_j < len(src_raw[max_index_i]):
                            tokens[i] = src_raw[max_index_i][max_index_j]
        return tokens

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert(len(translation_batch["gold_score"]) ==
               len(translation_batch["predictions"]))
        batch_size = batch.batch_size

        preds, pred_score, attn, gold_score, indices = list(zip(
            *sorted(zip(translation_batch["predictions"],
                        translation_batch["scores"],
                        translation_batch["attention"],
                        translation_batch["gold_score"],
                        batch.indices.data),
                    key=lambda x: x[-1])))

        # Sorting
        inds, perm = torch.sort(batch.indices.data)

        # here we use src to copy and replace_unk
        src = batch.src[0].data.index_select(0, perm)

        if self.has_tgt:
            tgt = batch.tgt.data.index_select(1, perm)
        else:
            tgt = None

        translations = []
        for b in range(batch_size):
            src_vocab = self.data.src_vocabs[inds[b]] \
                if self.data.src_vocabs else None
            src_raw = self.data.examples[inds[b]].src
            ex_raw = self.data.examples[inds[b].item()]

            pred_sents = [self._build_target_tokens(
                src[b] if src is not None else None,
                src_vocab, src_raw,
                preds[b][n], attn[b][n])
                for n in range(self.n_best)]
            gold_sent = None

            translation = Translation(src[b] if src is not None else None,
                                      src_raw, pred_sents,
                                      attn[b], pred_score[b], gold_sent,
                                      gold_score[b],
                                      ex_raw)
            translations.append(translation)

        return translations


class Translation(object):

    def __init__(self, src, src_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score,
                 ex_raw):
        #self.src = src
        #self.src_raw = src_raw
        self.pred_sents = pred_sents
        #self.attns = attn
        #self.pred_scores = pred_scores
        #self.gold_sent = tgt_sent
        #self.gold_score = gold_score
        self.ex_raw = ex_raw

    def log(self, sent_number):
        """
        Log translation.
        """

        output = '\nSENT {}: {}\n'.format(sent_number, self.src_raw)

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        output += 'PRED {}: {}\n'.format(sent_number, pred_sent)
        output += "PRED SCORE: {:.4f}\n".format(best_score)

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            output += 'GOLD {}: {}\n'.format(sent_number, tgt_sent)
            output += ("GOLD SCORE: {:.4f}\n".format(self.gold_score))
        if len(self.pred_sents) > 1:
            output += '\nBEST HYP:\n'
            for score, sent in zip(self.pred_scores, self.pred_sents):
                output += "[{:.4f}] {}\n".format(score, sent)

        return output
