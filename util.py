import logging
import os
from pathlib import Path
from re import S
import numpy as np
import torch
from matplotlib import pyplot as plt

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def _patch_noise_extend_to_img(noise, image_size=[3, 32, 32], patch_location='center'):
    c, h, w = image_size[0], image_size[1], image_size[2]
    mask = np.zeros((c, h, w), np.float32)
    x_len, y_len = noise.shape[1], noise.shape[2]

    if patch_location == 'center' or (h == w == x_len == y_len):
        x = h // 2
        y = w // 2
    elif patch_location == 'random':
        x = np.random.randint(x_len // 2, w - x_len // 2)
        y = np.random.randint(y_len // 2, h - y_len // 2)
    else:
        raise('Invalid patch location')

    x1 = np.clip(x - x_len // 2, 0, h)
    x2 = np.clip(x + x_len // 2, 0, h)
    y1 = np.clip(y - y_len // 2, 0, w)
    y2 = np.clip(y + y_len // 2, 0, w)
    mask[:, x1: x2, y1: y2] = noise
    return mask


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def log_display(epoch, global_step, time_elapse, **kwargs):
    display = 'epoch=' + str(epoch) + \
              '\tglobal_step=' + str(global_step)
    for key, value in kwargs.items():
        if type(value) == str:
            display = '\t' + key + '=' + value
        else:
            display += '\t' + str(key) + '=%.4f' % value
    display += '\ttime=%.2fit/s' % (1. / time_elapse)
    return display


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)

    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100/batch_size))
    return res

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_tag, dim = 1)
    correct_results_sum = (y_pred_tags == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100) 
    return acc


def save_model(filename, epoch, model, optimizer, scheduler, save_best=False, **kwargs):
    # Torch Save State Dict
    state = {
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None
    }
    for key, value in kwargs.items():
        state[key] = value
    if not isinstance(filename, str):
        torch.save(state, filename)
        filename = filename / Path('_best.pth')
    else:
        torch.save(state, filename + '.pth')
        filename += '_best.pth'    
    if save_best:
        torch.save(state, filename)
    return


def load_model(filename, model, optimizer, scheduler, **kwargs):
    # Load Torch State Dict
    filename = filename + '.pth'
    checkpoints = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoints['model_state_dict'])
    if optimizer is not None and checkpoints['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
    if scheduler is not None and checkpoints['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoints['scheduler_state_dict'])
    return checkpoints


def count_parameters_in_MB(model):
    return sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary_head" not in name)/1e6


def build_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)


def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def plot_loss_acc(epochs, losses_test, losses_train, acc_test, acc_train, dir_name, model_name):
    if not isinstance(dir_name, str):
        build_dirs(Path("./PLOTS") / dir_name)
    else:
        build_dirs(Path("./PLOTS/" + dir_name))

    if losses_test is not None and acc_test is not None:      
        fig, axs = plt.subplots(1,2, constrained_layout = True)

        axs[0].plot([x for x in range(len(losses_test))], losses_test, color = "blue", label = "Test/Validation Loss")
        axs[0].plot([x for x in range(len(losses_train))], losses_train, color = "red", label = "Training Loss")
        axs[0].legend()
        axs[0].set_xlabel("Epochs")
        axs[0].set_ylabel("Loss values")
        axs[0].set_title("Loss over Epochs")

        axs[1].plot([x for x in range(len(acc_test))], acc_test, color = "blue", label = "Test/Validation Accuracy")
        axs[1].plot([x for x in range(len(acc_train))], acc_train, color = "red", label = "Training Accuracy")
        axs[1].legend()
        axs[1].set_xlabel("Epochs")
        axs[1].set_ylabel("Accuracy (%)")
        axs[1].set_title("Accuracy over Epochs")

        plt.suptitle(model_name)

    else:
        fig, axs = plt.subplots(1,2, constrained_layout = True)

        axs[0].plot([x for x in range(len(losses_train))], losses_train, color = "red", label = "Training Loss")
        axs[0].legend()
        axs[0].set_xlabel("Epochs")
        axs[0].set_ylabel("Loss values")
        axs[0].set_title("Loss over Epochs")

        axs[1].plot([x for x in range(len(acc_train))], acc_train, color = "red", label = "Training Accuracy")
        axs[1].legend()
        axs[1].set_xlabel("Epochs")
        axs[1].set_ylabel("Accuracy (%)")
        axs[1].set_title("Accuracy over Epochs of")

        plt.suptitle(model_name)
        
    if not isinstance(dir_name, str):
        plt.savefig(Path("./PLOTS") / dir_name / Path(model_name + f"_epoch{epochs}.png"))
    else:
        plt.savefig(Path("./PLOTS/"+ dir_name + "/"+ model_name + f"_epoch{epochs}.png"))

def save_loss_acc(losses_test, losses_train, acc_test, acc_train, dir_name, model_name):
    build_dirs(dir_name)
    if not isinstance(dir_name, str):
        train_l = dir_name / Path("training_losses_" + model_name)
        test_l = dir_name / Path("test_losses_" + model_name)
        train_a = dir_name / Path("training_acc_" + model_name)
        test_a = dir_name / Path("test_acc_" + model_name)
    else:
        train_l = Path(dir_name) / Path("training_losses_" + model_name)
        test_l = Path(dir_name) / Path("test_losses_" + model_name)
        train_a = Path(dir_name) / Path("training_acc_" + model_name)
        test_a = Path(dir_name) / Path("test_acc_" + model_name)

    if losses_test is not None and acc_test is not None: 
        np.save(train_l, np.array(losses_train))
        np.save(test_l, np.array(losses_test))
        np.save(train_a, np.array(acc_train))
        np.save(test_a, np.array(acc_test))
    else:
        np.save(train_l, np.array(losses_train))
        np.save(train_a, np.array(acc_train))

def signed_absolute_maximum(list):
    orig_shape = np.array(list[0]).shape
    list = np.array([i.flatten() for i in list])

    list = list[np.argmax(np.abs(list), axis=0), np.arange(list.shape[1])].reshape(orig_shape)

    return list