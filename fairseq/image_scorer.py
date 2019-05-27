import torch


class ImageCTCLossScorer(object):
    """Scores the target for a given source image."""

    def __init__(self, tgt_dict, raw=False, strings=False):
        self.tgt_dict = tgt_dict
        self.blank_idx = tgt_dict.blank()
        self.raw = raw
        self.strings = strings

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Score a batch of images with best path decoding."""
        hypos_multi = []
        for model in models:
            model.eval()
            net_output = model(**sample['net_input'])
            lengths = torch.full((net_output.size(1),), net_output.size(0), dtype=torch.int32)
            _, net_output = net_output.max(2)
            net_output = net_output.transpose(1, 0).contiguous().reshape(-1)  # reshape
            tokens = self.decode(net_output, lengths)

            assert len(tokens) == len(sample['target'])
            hypos = []
            for token, target, name in zip(tokens, sample['target'], sample['image_name']):
                hypos.append({
                    'token': token,
                    'target': target.tolist(),
                    'name': name,
                })
            hypos_multi.append(hypos)

        return hypos_multi

    def decode(self, decoder_out, length):
        """Decode encoded labels back into strings.
        Args:
            decoder_out: torch.IntTensor [length_0 + length_1 + ...
                length_{n - 1}]: encoded labels.
            lenght: torch.IntTensor [n]: length of each labels.
        Raises:
            AssertionError: when the labels and its length does not match.
        Returns:
            labels (str or list of str): labels to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert decoder_out.numel() == length
            if self.raw:
                if self.strings:
                    return u''.join([self.tgt_dict.symbols[i] for i in decoder_out]).encode('utf-8')
                return decoder_out.tolist()
            else:
                decoder_out_non_blank = []
                for i in range(length):  # removing repeated characters and blank.
                    if (decoder_out[i] != self.blank_idx and (not (i > 0 and decoder_out[i - 1] == decoder_out[i]))):
                        if self.strings:
                            decoder_out_non_blank.append(self.tgt_dict.symbols[decoder_out[i]])
                        else:
                            decoder_out_non_blank.append(decoder_out[i].item())
                if self.strings:
                    return u''.join(decoder_out_non_blank).encode('utf-8')
                return decoder_out_non_blank
        else:  # batch mode
            assert decoder_out.numel() == length.sum()
            labels = []
            index = 0
            for i in range(length.numel()):
                idx_end = length[i]
                labels.append(self.decode(decoder_out[index:index + idx_end], torch.IntTensor([idx_end])))
                index += idx_end
            return labels
