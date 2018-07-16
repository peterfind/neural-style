# Copyright (c) 2015-2018 Anish Athalye. Released under GPLv3.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_VISIBLE_DEVICES']='1'
import numpy as np
import scipy.misc

from stylize import stylize

import math
from argparse import ArgumentParser

from PIL import Image

import utils
from collections import OrderedDict

# default arguments
CONTENT = './image/content/greatwall.jpg'
STYLE = ['./image/style/2-leonid.png']
OUTPUT = './image/output/output_test.png'
CHECKPOINT_OUTPUT = './image/mid_image/foo%s.jpg'
CONTENT_WEIGHT = 5e0
CONTENT_WEIGHT_BLEND = 1
STYLE_WEIGHT = 5e2
TV_WEIGHT = 1e2
STYLE_LAYER_WEIGHT_EXP = 1
LEARNING_RATE = 1e1
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
STYLE_SCALE = 1.0
ITERATIONS = 1000
VGG_PATH = '../vgg-models/imagenet-vgg-verydeep-19.mat'
POOLING = 'max'
def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content',
            dest='content', help='content image',
            metavar='CONTENT', default=CONTENT)
    parser.add_argument('--styles',
            dest='styles',
            nargs='+', help='one or more style images',
            metavar='STYLE', default=STYLE)
    parser.add_argument('--output',
            dest='output', help='output path',
            metavar='OUTPUT', default=OUTPUT)
    parser.add_argument('--iterations', type=int,
            dest='iterations', help='iterations (default %(default)s)',
            metavar='ITERATIONS', default=ITERATIONS)
    parser.add_argument('--print-iterations', type=int,
            dest='print_iterations', help='statistics printing frequency',
            metavar='PRINT_ITERATIONS')
    parser.add_argument('--checkpoint-output',
            dest='checkpoint_output', help='checkpoint output format, e.g. output%%s.jpg',
            metavar='OUTPUT', default=CHECKPOINT_OUTPUT)
    parser.add_argument('--checkpoint-iterations', type=int,
            dest='checkpoint_iterations', help='checkpoint frequency',
            metavar='CHECKPOINT_ITERATIONS')
    parser.add_argument('--width', type=int,
            dest='width', help='output width',
            metavar='WIDTH')
    parser.add_argument('--square_shape', action='store_true')
    parser.add_argument('--style_width', type=float, nargs='+')
    parser.add_argument('--network',
            dest='network', help='path to network parameters (default %(default)s)',
            metavar='VGG_PATH', default=VGG_PATH)
    parser.add_argument('--content-weight-blend', type=float,
            dest='content_weight_blend', help='content weight blend, conv4_2 * blend + conv5_2 * (1-blend) (default %(default)s)',
            metavar='CONTENT_WEIGHT_BLEND', default=CONTENT_WEIGHT_BLEND)
    parser.add_argument('--content-weight', type=float,
            dest='content_weight', help='content weight (default %(default)s)',
            metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    parser.add_argument('--style-weight', type=float,
            dest='style_weight', help='style weight (default %(default)s)',
            metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)
    parser.add_argument('--style-layer-weight-exp', type=float,
            dest='style_layer_weight_exp', help='style layer weight exponentional increase - weight(layer<n+1>) = weight_exp*weight(layer<n>) (default %(default)s)',
            metavar='STYLE_LAYER_WEIGHT_EXP', default=STYLE_LAYER_WEIGHT_EXP)
    parser.add_argument('--style-blend-weights', type=float,
            dest='style_blend_weights', help='style blending weights, weights between style images',
            nargs='+', metavar='STYLE_BLEND_WEIGHT')
    parser.add_argument('--tv-weight', type=float,
            dest='tv_weight', help='total variation regularization weight (default %(default)s)',
            metavar='TV_WEIGHT', default=TV_WEIGHT)
    parser.add_argument('--learning-rate', type=float,
            dest='learning_rate', help='learning rate (default %(default)s)',
            metavar='LEARNING_RATE', default=LEARNING_RATE)
    parser.add_argument('--beta1', type=float,
            dest='beta1', help='Adam: beta1 parameter (default %(default)s)',
            metavar='BETA1', default=BETA1)
    parser.add_argument('--beta2', type=float,
            dest='beta2', help='Adam: beta2 parameter (default %(default)s)',
            metavar='BETA2', default=BETA2)
    parser.add_argument('--eps', type=float,
            dest='epsilon', help='Adam: epsilon parameter (default %(default)s)',
            metavar='EPSILON', default=EPSILON)
    parser.add_argument('--initial',
            dest='initial', help='initial image',
            metavar='INITIAL')
    parser.add_argument('--initial-noiseblend', type=float,
            dest='initial_noiseblend', help='ratio of blending initial image with normalized noise (if no initial image specified, content image is used) (default %(default)s)',
            metavar='INITIAL_NOISEBLEND')
    parser.add_argument('--preserve-colors', action='store_true',
            dest='preserve_colors', help='style-only transfer (preserving colors) - if color transfer is not needed')
    parser.add_argument('--pooling',
            dest='pooling', help='pooling layer configuration: max or avg (default %(default)s)',
            metavar='POOLING', default=POOLING)
    parser.add_argument('--overwrite', action='store_true',
            dest='overwrite', help='write file even if there is already a file with that name')
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    utils.printoptions(options)

    if not os.path.isfile(options.network):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % options.network)

    content_image = utils.imread(options.content)
    style_images = [utils.imread(style) for style in options.styles]

    width = options.width
    if width is not None:  # resize content image
        if not options.square_shape:
            new_shape = (int(math.floor(float(content_image.shape[0]) /
                    content_image.shape[1] * width)), width)
        else:
            new_shape = (width, width)
        content_image = scipy.misc.imresize(content_image, new_shape)
    target_shape = content_image.shape
    for i in range(len(style_images)):  # resize style image
        if options.style_width is not None:
            style_images[i] = scipy.misc.imresize(style_images[i],
                (int(1.0 * style_images[i].shape[0] * options.style_width[i] / style_images[i].shape[1]), int(options.style_width[i])))

    style_blend_weights = options.style_blend_weights
    if style_blend_weights is None:
        # default is equal weights
        style_blend_weights = [1.0/len(style_images) for _ in style_images]
    else:
        total_blend_weight = sum(style_blend_weights)
        style_blend_weights = [weight/total_blend_weight
                               for weight in style_blend_weights]

    initial = options.initial
    if initial is not None:
        initial = scipy.misc.imresize(utils.imread(initial), content_image.shape[:2])
        # Initial guess is specified, but not noiseblend - no noise should be blended
        if options.initial_noiseblend is None:
            options.initial_noiseblend = 0.0
    else:
        # Neither inital, nor noiseblend is provided, falling back to random generated initial guess
        if options.initial_noiseblend is None:
            options.initial_noiseblend = 1.0
        if options.initial_noiseblend < 1.0:
            initial = content_image

    if options.checkpoint_output and "%s" not in options.checkpoint_output:
        parser.error("To save intermediate images, the checkpoint output "
                     "parameter must contain `%s` (e.g. `foo%s.jpg`)")

    # try saving a dummy image to the output path to make sure that it's writable
    if os.path.isfile(options.output) and not options.overwrite:
        raise IOError("%s already exists, will not replace it without the '--overwrite' flag" % options.output)
    try:
        utils.imsave(options.output, np.zeros((500, 500, 3)))
    except:
        raise IOError('%s is not writable or does not have a valid file extension for an image file' % options.output)

    for iteration, image, loss_dict in stylize(
        network=options.network,
        initial=initial,
        initial_noiseblend=options.initial_noiseblend,
        content=content_image,
        styles=style_images,
        preserve_colors=options.preserve_colors,
        iterations=options.iterations,
        content_weight=options.content_weight,
        content_weight_blend=options.content_weight_blend,
        style_weight=options.style_weight,
        style_layer_weight_exp=options.style_layer_weight_exp,
        style_blend_weights=style_blend_weights,
        tv_weight=options.tv_weight,
        learning_rate=options.learning_rate,
        beta1=options.beta1,
        beta2=options.beta2,
        epsilon=options.epsilon,
        pooling=options.pooling,
        print_iterations=options.print_iterations,
        checkpoint_iterations=options.checkpoint_iterations
    ):
        output_file = None
        combined_rgb = image
        if iteration is not None:  # iteration when optimize
            if options.checkpoint_output:
                output_file = options.checkpoint_output % iteration
                utils.imsave(output_file, combined_rgb)
        else:  # iteration of the last step
            output_file = options.output
            opt_image = Image.new('RGB', (image.shape[1], image.shape[0]), (200, 200, 200))
            opt_image = utils.drawdic(opt_image, OrderedDict(vars(options).items()+loss_dict.items()), hight=int((image.shape[0] - 5) / (len(vars(options)) + len(loss_dict))))
            rgb_with_message = np.append(combined_rgb, np.array(opt_image), axis=1)
            utils.imsave(output_file, combined_rgb)
            utils.imsave(os.path.join(os.path.dirname(output_file), 'mesg_'+os.path.basename(output_file)), rgb_with_message)

if __name__ == '__main__':
    print('\033[1;31;40m')
    print('Run neural_style.py')
    print('\033[0m')
    print('CUDA_VISIBLE_DEVICES = %s' %os.environ['CUDA_VISIBLE_DEVICES'])
    main()
    print('\n')
