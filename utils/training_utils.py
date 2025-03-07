import random
import shutil
import time

from torch.utils.tensorboard import SummaryWriter

from models import *
from utils.visualization import *


def get_tensorboard_logger_from_args(tensorboard_dir, reset_version=False):
    if reset_version:
        shutil.rmtree(os.path.join(tensorboard_dir))
    return SummaryWriter(log_dir=tensorboard_dir)


def get_optimizer_from_args(model, lr, weight_decay, **kwargs) -> torch.optim.Optimizer:
    return torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                             weight_decay=weight_decay)


def get_lr_schedule(optimizer):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_dir_from_args(root_dir, class_name, backbone, **kwargs):
    exp_name = f"{backbone}-{kwargs['dataset']}"

    if kwargs['OOM']:
        exp_name = f"{exp_name}-wi-OOM"
    else:
        exp_name = f"{exp_name}-wo-OOM"

    if kwargs['MOM']:
        exp_name = f"{exp_name}-wi-MOM"
    else:
        exp_name = f"{exp_name}-wo-MOM"

    csv_dir = os.path.join(root_dir, 'csv')
    csv_path = os.path.join(csv_dir, f"{exp_name}.csv")

    model_dir = os.path.join(root_dir, exp_name, 'models')
    img_dir = os.path.join(root_dir, exp_name, 'imgs')

    tensorboard_dir = os.path.join(root_dir, 'tensorboard', exp_name, class_name)
    logger_dir = os.path.join(root_dir, exp_name, 'logger', class_name)

    log_file_name = os.path.join(logger_dir,
                                 f'log_{time.strftime("%Y-%m-%d-%H-%I-%S", time.localtime(time.time()))}.log')

    model_name = f'{class_name}'

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(logger_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    handler = logging.FileHandler(log_file_name)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    logger.info(f"===> Root dir for this experiment: {logger_dir}")

    return model_dir, img_dir, tensorboard_dir, logger_dir, model_name, csv_path

def json_to_markdown_innested_list(json_dict, ind_level=0):

    markdown = ''
    for key, value in json_dict.items():
        if isinstance(value, dict):
            indentation = '\t' * ind_level
            markdown += f"{indentation}- **{key}**:\n"
            markdown += json_to_markdown_innested_list(value, ind_level + 1)
        else:
            indentation = '\t' * ind_level
            markdown += f"{indentation}- **{key}**: {value}\n"
    return markdown