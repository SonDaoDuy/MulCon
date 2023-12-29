from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """docstring for TrainOptions"""

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visualizer
        parser.add_argument('--print_freq', type=int, default=100,
                            help='frequency of showing training results on console')
        # network saving and loading params
        parser.add_argument('--save_latest_freq', type=int, default=100, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=1,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1,
                            help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--last_epoch', type=int, default=-1,
                            help='last epoch for pretrained with cosine lr')
        # parser.add_argument('--data_type', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--preprocess', type=str, default='none',
                            help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true',
                            help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--load_size', type=int, default=448, help='Image size for model feed forward, depend on the model of choice')

        # training params
        parser.add_argument('--niter', type=int, default=0, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=200,
                            help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.025, help='initial learning rate for adam')
        parser.add_argument('--lrp', type=float, default=0.1, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear',
                            help='learning rate policy. [linear | step | multi_step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--milestones', nargs='+',
                            help='step to decrease learning rate with multi step')
        parser.add_argument('--pretrain_folder', type=str, default=None, help='folder contain stage 1 of encoder in contrastive')
        parser.add_argument('--mask_model', type=str, default=None, help='folder contain stage 1 of encoder in contrastive')
        parser.add_argument('--train_nca', type=bool, default=False, help='train stage 1 of contrastive')
        parser.add_argument('--warm', action='store_true', help='warm up learning rate')
        parser.add_argument('--warm_epochs',type=int,default=5, help='warm up learning rate')
        #missing label setup
        parser.add_argument('--is_miss', action='store_true', help='declare missing label setup')
        parser.add_argument('--ema', action='store_true', help='declare missing label setup')
        parser.add_argument('--data_type', type=str, default='train',help='data type for the run [train | val| test]')
        parser.add_argument('--choose_size', type=float, default=0.1, help='declare choose label size for missing setup')
        self.isTrain = True
        return parser
