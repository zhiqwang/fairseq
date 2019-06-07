import os
import collections

import torch

from fairseq import options, tasks, checkpoint_utils, progress_bar, utils


def main(args):
    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., text recognition
    task = tasks.setup_task(args)
    # Load dataset
    task.load_dataset(args.gen_subset, combine=True, epoch=0)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    model_paths = args.path.split(':')
    models, _model_args = checkpoint_utils.load_model_ensemble(
        model_paths,
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Set dictionaries
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Initialize generator
    generator = task.build_generator(args)

    # Initialize data iterator
    itr = task.get_batch_iterator(
        dataset=task.datasets[args.gen_subset],
        max_tokens=args.max_tokens,
        max_positions=args.max_positions,
        max_sentences=args.max_sentences,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    progress = progress_bar.build_progress_bar(
        args, itr,
        prefix='inference on \'{}\' subset'.format(args.gen_subset),
        no_progress_bar='simple'
    )

    stats = collections.OrderedDict()
    num_correct = [0 for _ in range(len(model_paths))]
    num_verified = [0 for _ in range(len(model_paths))]
    wrong_lists = [[] for _ in range(len(model_paths))]

    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        hypos_multi = task.inference_step(generator, models, sample)
        for i, hypos in enumerate(hypos_multi):
            num_verified[i] += len(hypos)
            for hypo in hypos:
                target = hypo['target']
                token = hypo['token']
                if token == target:
                    num_correct[i] += 1
                else:
                    wrong_lists[i].append({
                        'target': target,
                        'token': token,
                        'name': hypo['name'],
                    })

        for i in range(len(model_paths)):
            stats['acc{}'.format(i + 1)] = num_correct[i] / num_verified[i]
        progress.log(stats, tag='accuracy')
    progress.print(stats, tag='accuracy')

    # print wrong pair
    for i in range(len(model_paths)):
        print('| error pair in {} | count {}\n'.format(model_paths[i], len(wrong_lists[i])))
        for wrong_pair in wrong_lists[i]:
            image_name = os.path.basename(wrong_pair['name']).split('.')[0]
            print('name: {}'.format(image_name))
            print('target: {}'.format(tgt_dict.string(wrong_pair['target'])))
            print('tokens: {}\n'.format(tgt_dict.string(wrong_pair['token'])))


def cli_main():
    parser = options.get_generation_parser(interactive=True)
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
