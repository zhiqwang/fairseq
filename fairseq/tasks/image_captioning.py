import os
import torchvision.transforms as transforms

from fairseq.tasks import FairseqTask, register_task
from fairseq.data import ImageCaptioningDataset
from fairseq.data.image_captioning_dictionary import ImageCaptioningDictionary, Dictionary


@register_task('image_captioning')
class ImageCaptioningTask(FairseqTask):
    """
    Train a image captioning model.

    Args:
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target text
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='path to data directory')
        parser.add_argument('--max-positions', default=1024, type=int,
                            help='max input length')
        parser.add_argument('--height', type=int, default=32,
                            help='image height size used for training (default: 32)')
        parser.add_argument('--width', type=int, default=200,
                            help='image width size used for training (default: 200)')
        parser.add_argument('--keep-ratio', action='store_true',
                            help='keep image size ratio when training')
        parser.add_argument('--not-pin', dest='pin-memory', action='store_false',
                            help='training using pined memory')
        # fmt: on

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        use_ctc_loss = True if args.criterion == 'ctc_loss' else False
        tgt_dict = cls.load_dictionary(os.path.join(args.data, 'dict.txt'), use_ctc_loss)
        print('| target dictionary: {} types'.format(len(tgt_dict)))

        return cls(args, tgt_dict)

    def __init__(self, args, tgt_dict):
        super().__init__(args)
        self.tgt_dict = tgt_dict

    @classmethod
    def load_dictionary(cls, filename, use_ctc_loss):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        if use_ctc_loss:
            return ImageCaptioningDictionary.load(filename)
        return Dictionary.load(filename)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        # Read input images and targets
        image_names = []
        targets = []
        target_lengths = []
        image_root = os.path.join(self.args.data, 'images')
        label_path = os.path.join(self.args.data, '{}.txt'.format(split))
        with open(label_path, mode='r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split()
                image_names.append(os.path.join(image_root, line[0]))
                targets.append(line[1:])
                target_lengths.append(len(line[1:]))

        assert len(image_names) == len(targets) == len(target_lengths)
        print('| {} {} {} images'.format(self.args.data, split, len(image_names)))

        mean = [0.396, 0.576, 0.562]
        std = [0.154, 0.128, 0.130]

        image_size = self.args.height if self.args.keep_ratio else (self.args.height, self.args.width)

        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        shuffle = True if split == 'train' else False
        append_eos_to_target = False if self.args.criterion == 'ctc_loss' else True
        use_ctc_loss = True if self.args.criterion == 'ctc_loss' else False
        self.datasets[split] = ImageCaptioningDataset(
            image_names, targets, self.tgt_dict, tgt_sizes=target_lengths,
            shuffle=shuffle, transform=transform, use_ctc_loss=use_ctc_loss,
            input_feeding=True, append_eos_to_target=append_eos_to_target,
        )

    def build_generator(self, args):
        from fairseq.image_scorer import ImageCTCLossScorer
        if args.criterion == 'ctc_loss':
            return ImageCTCLossScorer(self.target_dictionary)

    def max_positions(self):
        """Return the max input length allowed by the task."""
        # The source should be less than *args.max_positions* and
        # In order to use `CuDNN`, the "target" has max length 256,
        return (self.args.max_positions, 256)

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
