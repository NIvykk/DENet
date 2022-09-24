import argparse
import os
import platform
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytz
import torch
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter


class BaseOptions():
    # This class defines options used during both training and test time.
    def __init__(self, ):
        parser = argparse.ArgumentParser(
            description="Options for training and test")
        # basic parameters
        parser.add_argument(
            '--project',
            type=str,
            default='',
            help='project name of the experiment. used to makedir [run_dir]')
        parser.add_argument(
            '--name',
            type=str,
            default='yolov3',
            help='name of the experiment. used to makedir [run_dir]')
        parser.add_argument('--data',
                            type=str,
                            default='./data/voc0712.yaml',
                            help='data.yaml path')
        parser.add_argument('--data_mode',
                            type=str,
                            default='normal',
                            choices=['normal', 'train_paired'],
                            help='dataset mode')
        parser.add_argument('--hyp',
                            type=str,
                            default='./hyp/hyp.voc.scratch.yaml',
                            help='hyperparameters path')
        parser.add_argument('--model',
                            type=str,
                            default='yolov3.YOLOv3',
                            help='model')
        parser.add_argument('--batch_size',
                            '--bs',
                            type=int,
                            default=2,
                            help='total batch size for all GPUs')
        parser.add_argument('--checkpoint',
                            type=str,
                            default='',
                            help='checkpoint path')
        parser.add_argument('--run_dir',
                            type=str,
                            default='./runs',
                            help='logs dirs,do not need modify,auto process')
        parser.add_argument('--device', type=str, default='', help='device')
        parser.add_argument('--cpu',
                            action='store_true',
                            help='only use cpu to run ?')
        parser.add_argument('--filter_cls',
                            action='store_true',
                            help='only detect specific classes')
        parser.add_argument('--tb_writer',
                            type=bool,
                            default=True,
                            help='use tensorboard logger ?')
        parser.add_argument('--verbose',
                            action='store_true',
                            help='verbose report')
        parser.add_argument('--conf_thres',
                            type=float,
                            default=0.001,
                            help='object confidence threshold')
        parser.add_argument('--nms_thres',
                            type=float,
                            default=0.5,
                            help='IOU threshold for NMS')
        # get parser
        self.parser = parser

    def init(self):
        # run_log
        self.make_run_dir(self.opt)
        # stdout/stderr
        self.redirect_stdout_stderr(log_dir=self.opt.run_dir,
                                    stdout=True,
                                    stderr=True)
        # seeds
        self.init_seeds(seed=100)
        # device
        self.select_device(self.opt)
        # env info
        self.print_env_info(self.opt)
        # opt
        self.print_options(self.parser, self.opt)
        # Tensorboard loggers
        self.init_tb_writer(self.opt)
        # hyp.yaml、data.yaml
        self.save_config_to_file(self.opt)

        return self.opt

    @staticmethod
    def save_config_to_file(opt):
        from pathlib import Path

        import yaml
        run_dir = Path(opt.run_dir)
        with open(opt.hyp) as f:
            hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyp
        with open(run_dir / 'hyp.yaml', 'w') as f:  # save hyp to file
            yaml.dump(hyp, f, sort_keys=False)
        with open(opt.data) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)  # load hyp
        with open(run_dir / 'data.yaml', 'w') as f:  # save hyp to file
            yaml.dump(data, f, sort_keys=False)

    @staticmethod
    def make_run_dir(opt):
        # run_dir: ./runs/[opt.task]/[time]_[opt.name]
        time = datetime.now(
            pytz.timezone('Asia/Shanghai')).strftime(r"%Y-%m-%d-%H-%M-%S")
        opt.run_start_time = time
        opt.run_dir = str(
            Path(opt.run_dir) / opt.task / opt.project / f"{time}_{opt.name}")
        os.makedirs(opt.run_dir, exist_ok=True)

    @staticmethod
    def redirect_stdout_stderr(log_dir='./', stdout=True, stderr=True):
        """
        redirect_stdout_stderr(log_dir='./', stdout=True, stderr=True)
        """
        if stdout:
            sys.stdout = Logger(log_dir, terminal=sys.stdout)
        if stderr:
            sys.stderr = Logger(log_dir, terminal=sys.stderr)

    @staticmethod
    def select_device(opt):
        if not opt.cpu:
            if torch.cuda.is_available():
                if torch.cuda.device_count() > 1:
                    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
                opt.device = torch.device("cuda")
        else:
            opt.device = torch.device("cpu")

    @staticmethod
    def init_tb_writer(opt):
        # init Tensorboard loggers
        opt.tb_writer = SummaryWriter(opt.run_dir) if opt.tb_writer else None
        print(
            f'Start Tensorboard with "tensorboard --logdir {os.path.dirname(opt.run_dir)}", view at http://localhost:6006/\n'
        )

    @staticmethod
    def init_seeds(seed=100):
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        os.environ["PYTHONHASHSEED"] = str(seed)  # hash
        torch.manual_seed(seed)  # CPU
        torch.cuda.manual_seed(seed)  # GPU
        torch.cuda.manual_seed_all(seed)  # GPU
        # Reduce randomness (may be slower on Tesla GPUs) # https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.enabled = True  # cudnn
        torch.backends.cudnn.benchmark = True  # benchmark
        torch.backends.cudnn.deterministic = False  # cudnn
        # benchmark=false, deterministic=true ==> slower and more reproducible

    @staticmethod
    def print_options(parser, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [run_dir] / opt.txt
        """
        options_info = ""
        options_info += "#-----------------------------------Options-------------------------------------------\n"
        for k, v in (vars(opt).items()):
            comment = ''
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            options_info += '{}: {}{}\n'.format(str(k), str(v), comment)
        options_info += "#-------------------------------------------------------------------------------------\n"

        print(options_info)
        # save to the file
        file_name = Path(opt.run_dir) / "opt.yaml"
        with open(file_name, 'wt') as opt_file:
            opt_file.write(options_info)

    @staticmethod
    def print_env_info(opt):
        env_info = "#------------------------------Environment information---------------------------------\n"
        env_info += f"Date: {opt.run_start_time}\n"  # date
        env_info += f"OS: {platform.system()} {platform.release()}\n"  # os version
        env_info += f"Python: {platform.python_version()}\n"  # Python version
        env_info += f"Pytorch: {torch.__version__}\n"  # PyTorch version
        if not opt.cpu:
            if torch.cuda.is_available():
                env_info += f"CUDA: True ({torch.version.cuda})\n"  # CUDA version
                if cudnn.is_acceptable(torch.tensor(1.).cuda()):
                    env_info += f"cuDNN: True ({torch.backends.cudnn.version()})\n"
                    # cuDNN version
                gpu_info = torch.cuda.get_device_properties(0)
                env_info += f"GPU: {gpu_info.name} ({gpu_info.total_memory // (1024**2)} MB)\n"  # GPU type
        env_info += "#-------------------------------------------------------------------------------------\n"

        print(env_info)
        file_name = Path(opt.run_dir) / "env.yaml"
        with open(file_name, 'wt') as env_file:
            env_file.write(env_info)


class Logger(object):
    def __init__(self, log_dir='./', terminal=sys.stdout):
        os.makedirs(log_dir, exist_ok=True)
        log_file = Path(log_dir) / 'run.log'
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log = open(self.log_file, "a")
        self.log.write(message)
        self.log.close()

    def flush(self):
        # sys.terminal.flush()
        pass
