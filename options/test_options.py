from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.
    It also includes shared options defined in BaseOptions.
    """

    def __init__(self, ):
        super(TestOptions, self).__init__()
        parser = self.parser
        # testing parameters
        parser.add_argument('--task',
                            type=str,
                            default='val',
                            choices=['val', 'test'],
                            help='kind of task')
        parser.add_argument('--img_size_test',
                            nargs='+',
                            type=int,
                            default=[544, 544],
                            help='[w,h] image size')

        # -----------init opt-----------#
        self.opt = parser.parse_args()


if __name__ == "__main__":
    """
    from options.test_options import TestOptions
    opt = TestOptions().init()   # get test options
    """
    opt = TestOptions().init()