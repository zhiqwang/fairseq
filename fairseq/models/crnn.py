import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
import fairseq.modules.convolution as convolution


# pretrained features
FEATURES = {}

# output dimensionality for supported architectures
OUTPUT_DIM = {
    'resnet_cifar': 512,
    'densenet_cifar': 342,
    'densenet121': 384,
    'mobilenetv2_cifar': 1280,
    'shufflenetv2_cifar': 1024,
}


@register_model('text_recognition')
class CRNNModel(FairseqEncoderDecoderModel):
    """
    CRNN model from `"An End-to-End Trainable Neural Network for Image-based
    Sequence Recognition and Its Application to Scene Text Recognition" (Shi, et al, 2015)
    <https://arxiv.org/abs/1507.05717>`_.

    Args:
        encoder (CRNNEncoder): the encoder
        decoder (CRNNDecoder): the decoder

    The CRNN model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.crnn_parser
        :prog:
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--backbone', default='densenet_cifar',
                            help='CNN backbone architecture. (default: densenet_cifar)')
        parser.add_argument('--pretrained', action='store_true', help='pretrained')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='number of decoder layers')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)
        encoder = CRNNEncoder(args.backbone, args.pretrained)
        decoder = CRNNDecoder(
            dictionary=task.target_dictionary,
            embed_dim=encoder.embed_dim,
            num_layers=args.decoder_layers,
            bidirectional=True,
        )
        return cls(encoder, decoder)

    def forward(self, image):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source image through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., input feeding/teacher
        forcing) to the decoder to produce the next outputs::

            encoder_out = self.encoder(image)
            return self.decoder(encoder_out)

        Args:
            image (Tensor): tokens in the source image of shape
                `(batch, channel, img_h, img_w)`

        Returns:
            the decoder's output, typically of shape `(tgt_len, batch, vocab)`
        """
        encoder_out = self.encoder(image)
        decoder_out = self.decoder(encoder_out)

        return decoder_out


class CRNNEncoder(FairseqEncoder):
    """CRNN encoder."""
    def __init__(self, backbone, pretrained):
        super(FairseqEncoder, self).__init__()
        self.embed_dim = OUTPUT_DIM[backbone]
        self.position_embeddings = nn.Embedding(self.max_positions(), self.embed_dim)
        self.features = nn.Sequential(*self.cnn_layers(backbone, pretrained))
        self.avgpool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, images):
        """
        Args:
            images (Tensor): tokens in the source images of shape
                `(bsz, channel, img_h, img_w)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, bsz, embed_dim)`
        """
        # images -> features
        x = self.features(images)  # bsz x embed_dim x H' x W', where W' stands for `seq_len`
        # features -> pool -> flatten
        x = self.avgpool(x)
        x = x.permute(3, 0, 1, 2).view(x.size(3), x.size(0), -1)  # seq_len x bsz x embed_dim

        if self.position_embeddings is not None:
            positions = torch.arange(len(x), device=x.device).unsqueeze(-1)
            x = x + self.position_embeddings(positions).expand_as(x)

        return {
            'encoder_out': x,  # T x B x C
        }

    def max_positions(self):
        """Maximum sequence length supported by the encoder."""
        return 128

    @staticmethod
    def cnn_layers(backbone, pretrained):
        """CNN backbone for the CRNN Encoder."""

        # loading network
        if pretrained:
            if backbone not in FEATURES:
                # initialize with network pretrained on imagenet in pytorch
                net_in = getattr(convolution, backbone)(pretrained=True)
            else:
                # initialize with random weights, later on we will fill features with custom pretrained network
                net_in = getattr(convolution, backbone)(pretrained=False)
        else:
            # initialize with random weights
            net_in = getattr(convolution, backbone)(pretrained=False)

        # initialize features
        # take only convolutions for features,
        # always ends with ReLU to make last activations non-negative
        if backbone.startswith('resnet'):
            features = list(net_in.children())[:-2]
        elif backbone.startswith('densenet'):
            features = list(net_in.features.children())
            features.append(nn.ReLU(inplace=True))
        elif backbone.startswith('mobilenetv2'):
            features = list(net_in.children())[:-2]
        elif backbone.startswith('shufflenetv2'):
            features = list(net_in.children())[:-2]
        else:
            raise ValueError('Unsupported or unknown architecture: {}!'.format(backbone))

        return features


class CRNNDecoder(FairseqDecoder):
    """CRNN decoder."""
    def __init__(
        self, dictionary, embed_dim, hidden_size=512,
        num_layers=1, attention=None, bidirectional=False,
    ):
        super().__init__(dictionary)
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        encoder_hidden_size = hidden_size * num_layers
        if bidirectional:
            encoder_hidden_size *= 2
        self.encoder_hidden = nn.Linear(embed_dim, encoder_hidden_size)
        self.encoder_cell = nn.Linear(embed_dim, encoder_hidden_size)

        self.rnn = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )

        self.fc = nn.Linear(hidden_size, len(dictionary))

    def forward(self, encoder_out):
        # encoder_out -> decoder
        x = encoder_out['encoder_out']  # seq_len x bsz x embed_dim

        (h0, c0) = self.init_hidden(x)

        out, _ = self.rnn(x, (h0, c0))
        # Sum bidirectional RNN outputs
        if self.bidirectional:
            out = out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:]
        out = self.fc(out)

        return out

    def init_hidden(self, x):
        mean = torch.mean(x, dim=0)  # bsz x embed_dim

        h0 = self.encoder_hidden(mean)  # bsz x encoder_hidden_size
        h0 = h0.view(mean.size(0), -1, self.hidden_size)
        h0 = h0.transpose(0, 1).contiguous()
        h0 = F.relu6(h0, inplace=True)

        c0 = self.encoder_cell(mean)  # bsz x encoder_hidden_size
        c0 = c0.view(mean.size(0), -1, self.hidden_size)
        c0 = c0.transpose(0, 1).contiguous()
        c0 = F.relu6(c0, inplace=True)

        return (h0, c0)

    def get_normalized_probs(self, net_output, log_probs, sample):
        """Get normalized probabilities (or log probs) from a net's output."""
        if log_probs:
            return utils.log_softmax(net_output, dim=2)
        else:
            return utils.softmax(net_output, dim=2)


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


@register_model_architecture('text_recognition', 'crnn')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.backbone = getattr(args, 'backbone', 'densenet_cifar')
    args.pretrained = getattr(args, 'pretrained', False)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
