from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        # parser.add_argument('--save', type=str, default='/mnt/Data/Son/sosc_new/segan_amodal_end2end/', help='save folder')
        parser.add_argument('--num_test', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--data_type', type=str, default='test',help='data type for the run [train | val| test]')
        parser.add_argument('--niter_decay', type=int, default=200,help='data type for the run [train | val| test]')
        parser.add_argument('--preprocess', type=str, default='none',
                            help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true',
                            help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--load_size', type=int, default=448, help='Image size for model feed forward, depend on the model of choice')
        parser.add_argument('--train_nca', type=bool, default=False, help='Image size for model feed forward, depend on the model of choice')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--ema', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--is_miss', action='store_true', help='declare missing label setup')
        # rewrite devalue values
        # parser.set_defaults(model='test')
        self.isTrain = False
        return parser
