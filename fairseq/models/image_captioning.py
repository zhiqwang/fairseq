import math

import torch
import torch.nn as nn

from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)

import fairseq.modules.convolution as convolution
from fairseq.models.lstm import LSTM

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


@register_model('image_captioning')
class ImageCaptioningModel(FairseqEncoderDecoderModel):
    """
    CRNN model from `"An End-to-End Trainable Neural Network for Image-based
    Sequence Recognition and Its Application to Scene Text Recognition" (Shi, et al, 2015)
    <https://arxiv.org/abs/1507.05717>`_.

    Args:
        encoder (ImageCaptioningEncoder): the encoder
        decoder (ImageCaptioningDecoder): the decoder

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
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-hidden-size', type=int, metavar='N',
                            help='decoder hidden size')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='number of decoder layers')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--no-token-rnn', default=False, action='store_true',
                            help='if set, disables rnn layer')
        parser.add_argument('--no-token-crf', default=False, action='store_true',
                            help='if set, disables conditional random fields')
        parser.add_argument('--decoder-bidirectional', action='store_true',
                            help='make all layers of decoder bidirectional')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)
        encoder = ImageCaptioningEncoder(
            args=args,
        )
        decoder = ImageCaptioningDecoder(
            dictionary=task.target_dictionary,
            embed_dim=encoder.embed_dim,
            bidirectional=args.decoder_bidirectional,
            num_layers=args.decoder_layers,
            no_token_rnn=args.no_token_rnn,
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


class ImageCaptioningEncoder(FairseqEncoder):
    """Image captioning encoder."""
    def __init__(self, args):
        super(FairseqEncoder, self).__init__()
        self.embed_dim = OUTPUT_DIM[args.backbone]
        self.features = nn.Sequential(*self.cnn_layers(args.backbone, args.pretrained))
        self.avgpool = nn.AdaptiveAvgPool2d((1, None))

        self.embed_positions = PositionalEncoding(
            embedding_dim=self.embed_dim,
            num_embeddings=self.max_positions(),
        ) if not args.no_token_positional_embeddings else None

        self.decoder_need_rnn = not args.no_token_rnn
        if self.decoder_need_rnn:
            hidden_size = args.decoder_hidden_size

            init_layers = args.decoder_layers
            if args.decoder_bidirectional:
                init_layers *= 2

            self.init_hidden_w = nn.Parameter(
                torch.rand(init_layers, self.embed_dim, hidden_size)
            )  # init_layers x embed_dim x hidden_size
            self.init_hidden_b = nn.Parameter(
                torch.rand(init_layers, 1, hidden_size)
            )  # init_layers x 1 x hidden_size
            self.init_cell_w = nn.Parameter(torch.rand_like(self.init_hidden_w))
            self.init_cell_b = nn.Parameter(torch.rand_like(self.init_hidden_b))

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

        final_hiddens, final_cells = None, None

        if self.decoder_need_rnn:
            final_hiddens, final_cells = self.init_hidden(x)

        if self.embed_positions is not None:
            x += self.embed_positions(x)

        return {
            'encoder_out': (x, final_hiddens, final_cells),
            'encoder_padding_mask': None,
        }

    def init_hidden(self, x):
        mean = torch.mean(x, dim=0)  # bsz x embed_dim

        h0 = mean @ self.init_hidden_w + self.init_hidden_b  # init_layers x bsz x hidden_size
        h0 = torch.tanh(h0)

        c0 = mean @ self.init_cell_w + self.init_cell_b  # init_layers x bsz x hidden_size
        c0 = torch.tanh(c0)

        return (h0, c0)

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


class ImageCaptioningDecoder(FairseqIncrementalDecoder):
    """CRNN decoder."""
    def __init__(
        self, dictionary, embed_dim, hidden_size=512,
        bidirectional=True, num_layers=2, no_token_rnn=False,
    ):
        super().__init__(dictionary)
        self.need_rnn = not no_token_rnn
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers

        self.rnn = LSTM(
            input_size=embed_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
        ) if self.need_rnn else None

        hidden_size = self.hidden_size if self.need_rnn else embed_dim
        self.classifier = nn.Linear(hidden_size, len(dictionary))

    def forward(self, encoder_out):
        # encoder_out -> decoder
        encoder_out = encoder_out['encoder_out']  # seq_len x bsz x embed_dim

        # get outputs from encoder
        encoder_outs, encoder_hiddens, encoder_cells = encoder_out[:3]

        if self.rnn is not None:
            x, _ = self.rnn(encoder_outs, (encoder_hiddens, encoder_cells))
            # Sum bidirectional RNN xputs
            if self.bidirectional:
                x = x[:, :, :self.hidden_size] + x[:, :, self.hidden_size:]

        x = self.classifier(x)

        return x

    def get_normalized_probs(self, net_output, log_probs, sample):
        """Get normalized probabilities (or log probs) from a net's output."""
        if log_probs:
            return utils.log_softmax(net_output, dim=2)
        else:
            return utils.softmax(net_output, dim=2)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, embedding_dim, num_embeddings=128):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(num_embeddings, embedding_dim)  # embed_num x embed_dim
        position = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1)
        emb = math.log(10000.0) / embedding_dim
        emb = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float) * -emb)
        pe[:, 0::2] = torch.sin(position * emb)
        pe[:, 1::2] = torch.cos(position * emb)
        self.register_buffer('pe', pe)

    def forward(self, x):
        out = self.pe[:x.size(0)]  # seq_len x embed_dim
        out = out.unsqueeze(1)
        out = out.expand_as(x)
        return out


@register_model_architecture('image_captioning', 'image_captioning')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.backbone = getattr(args, 'backbone', 'densenet_cifar')
    args.pretrained = getattr(args, 'pretrained', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', args.decoder_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 2)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.no_token_rnn = getattr(args, 'no_token_rnn', False)
    args.no_token_crf = getattr(args, 'no_token_crf', True)
    args.decoder_bidirectional = getattr(args, 'decoder_bidirectional', False)


@register_model_architecture('image_captioning', 'decoder_crnn')
def decoder_crnn(args):
    args.decoder_bidirectional = getattr(args, 'decoder_bidirectional', True)
    base_architecture(args)


@register_model_architecture('image_captioning', 'decoder_attention')
def decoder_attention(args):
    args.no_token_rnn = getattr(args, 'no_token_rnn', False)
    args.no_token_crf = getattr(args, 'no_token_crf', False)
    base_architecture(args)
