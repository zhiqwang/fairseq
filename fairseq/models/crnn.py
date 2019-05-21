import torch.nn as nn

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

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--backbone', default='densenet_cifar',
                            help='CNN backbone architecture. (default: densenet_cifar)')
        parser.add_argument('--pretrained', action='store_true', help='pretrained')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)
        encoder = CRNNEncoder(args.backbone, args.pretrained)
        decoder = CRNNDecoder(task.target_dictionary, encoder.hidden_size)
        return cls(encoder, decoder)

    def __init__(self, encoder, decoder):
        super(FairseqEncoderDecoderModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, image):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., input feeding/teacher
        forcing) to the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing

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
        self.features = nn.Sequential(*self.cnn_backbone(backbone, pretrained))
        self.avgpool = nn.AdaptiveAvgPool2d((1, None))
        self.hidden_size = OUTPUT_DIM[backbone]

    def forward(self, images):
        # images -> features
        x = self.features(images)
        # features -> pool -> flatten
        x = self.avgpool(x)
        x = x.permute(3, 0, 1, 2).view(x.size(3), x.size(0), -1)
        return {
            'encoder_out': x,  # T x B x C
        }

    @staticmethod
    def cnn_backbone(backbone, pretrained):
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
    def __init__(self, dictionary, hidden_size):
        super().__init__(dictionary)
        self.classifier = nn.Linear(hidden_size, len(dictionary))

    def forward(self, encoder_out):
        # encoder_out -> decoder
        x = self.classifier(encoder_out['encoder_out'])

        return x

    def get_normalized_probs(self, net_output, log_probs, sample):
        """Get normalized probabilities (or log probs) from a net's output."""
        if log_probs:
            return utils.log_softmax(net_output, dim=2)
        else:
            return utils.softmax(net_output, dim=2)


@register_model_architecture('text_recognition', 'crnn')
def base_architecture(args):
    args.backbone = getattr(args, 'backbone', 'densenet_cifar')
    args.pretrained = getattr(args, 'pretrained', False)
