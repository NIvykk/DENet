from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """ This class includes training options.
        It also includes shared options defined in BaseOptions.
    """
    def __init__(self, ):
        super(TrainOptions, self).__init__()
        parser = self.parser
        # training parameters
        parser.add_argument('--img_size_train',
                            nargs='+',
                            type=int,
                            default=[640, 352],
                            help='[w,h] image size')
        parser.add_argument('--img_size_test',
                            nargs='+',
                            type=int,
                            default=[640, 352],
                            help='[w,h] image size')
        parser.add_argument('--epochs',
                            type=int,
                            default=300,
                            help='number of epochs')
        parser.add_argument('--freeze_layers',
                            nargs='+',
                            type=str,
                            default=[],
                            help='freeze_layers: backbone, neck, Detect')
        parser.add_argument('--noamp',
                            action='store_true',
                            help='disable amp training')
        parser.add_argument('--resume',
                            action='store_true',
                            help='resume training from checkpoint')
        parser.add_argument('--multi_scale',
                            action='store_true',
                            help='multi_scale training')
        parser.add_argument('--noautoanchor',
                            action='store_true',
                            help='disable autoanchor check')
        parser.add_argument('--task',
                            type=str,
                            default='train',
                            choices=['train'],
                            help='kind of task')
        # -----------init opt-----------#
        self.opt = parser.parse_args()


if __name__ == "__main__":
    """
    from options.train_options import TrainOptions
    opt = TrainOptions().init()   # get training options
    """

    opt = TrainOptions().init()
