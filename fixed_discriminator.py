import random
from torch.autograd import Variable
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import numpy as np
from sklearn.decomposition import PCA
import argparse
from tqdm import tqdm
from models.CNN_detector import CNN_detector, LeNet
from models.wideresnet import WideResNet
import ssl
from models.generatorResNet import GeneratorResnet, GeneratorResnet_P_Ensemble, GeneratorResnetEnsemble
from models.ResNet import ResNet18
from models.Linear import linearModel
import util
from partialCusomDataset import perturbDataset
from models.PreTrainedModels import ResNet, get_model


parser = argparse.ArgumentParser(description='General ShortcutGen training run')
parser.add_argument('--b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', default=0.0002, type=float,
                    help='Learning rate of optimizer')
parser.add_argument('--beta_1', default=0.5, type=float,
                    help='First Beta parameter of the ADAM optimizer (Default : 0.5) ')
parser.add_argument('--dataset', type= str, default = "CIFAR10",
                    help="Which dataset to train on (default: CIFAR10)")
parser.add_argument('--resolution', default = 32, type = int,
                    help = "Resolution if ImageNet is used (Default: 32)")
parser.add_argument('--epochs', type = int, default = 25, 
                    help = "amount of epochs performed (defult = 25)")
parser.add_argument('--discr_path', type =str, default = "./PreTrainedDiscriminator/Default_WideResNet/WideResNethigh_train.pt",
                    help = "String that corresponds with the path to the pre-trained classifier that will be used")
parser.add_argument('--exp_name', type =str, default = "Only_AdvGanLoss",
                    help = "String that will be used as name for the specific directory the results will be saved in")
parser.add_argument('--load_gen', type =str, default = None,
                    help = "String that corresponds with an generator model that already had some training")
parser.add_argument('--ensemble_numbers', type =int, default = None, nargs="+",
                    help = "3 numbers that if given represent the ensemble of generators to use (e.g. 2 1 3 will give a enseemble of 2 postaug generator, 1 aug generator and 3 no aug generators). If defaulted to None, normal generator will be used")
parser.add_argument('--dir_name', type =str, default = "FixedDiscriminator",
                    help = "String that will be used as name for the directory that the results will be saved in; This is not equal to the exp_name")
parser.add_argument('--plot', type = bool, default = False, 
                    help = "If true plots the training and validation losses and accuracy over epochs")
parser.add_argument('--Random_20', type = int, default = None, 
                    help = "If not None will be the seed for the generator splitting the trainingset in 20-80")
parser.add_argument('--noclamp', type = bool, default = False, 
                    help = "If true no clamping will be performed in the 0-1 range. (Default: False)")
parser.add_argument('--noaugment', type = bool, default = False, 
                    help = "If true no augmentation will be used (default: False)")
parser.add_argument('--post_augment', type = bool, default = False, 
                    help = "If true augmentation will be used between generator and classifier (default: False)")
parser.add_argument('--grayaug', type = bool, default = False, 
                    help = "If true GRAYAUG preprocessing will be used (default: False)")
parser.add_argument('--init_discriminator', type = bool, default = False, 
                    help = "Use a static random initialized dicriminator")
parser.add_argument('--saveLists', type = bool, default = False, 
                    help = "If true save the numpy arrays containing the loss and accuracy for training and validation")
parser.add_argument('--discr_seed', type = int, default = None, 
                    help = "If not none, will be used to set the seed just of the classifier used to train against")
parser.add_argument('--transform_seed', type = int, default = None, 
                    help = "If not none, will be used to set the seed just before the start of the first epoch, and as such make the random augmentation seeded")
parser.add_argument('--model', type =str, default = "ResNet18",
                    help = "Type of victim model (options: ResNet18 and Linear)")               
# parser.add_argument('--augment', type = bool, default = False, 
#                     help = "True: Augment the training data; False: No augmentation is used (default : False)")
parser.add_argument('--saveLastModel', type = bool, default = False, 
                    help = "True: Last model will be saved (additionally); False: Last model will not be saved (default: False)")
parser.add_argument('--epsilon', default=8, type=float, help='perturbation')

ssl._create_default_https_context = ssl._create_unverified_context

args = parser.parse_args()
eps = args.epsilon/255
dir_name = Path(args.dir_name + "/" + args.exp_name)
model_name = "GenResNet_OnlyAdvGAN"
dataset_path = Path("../datasets")
discriminator_path = args.discr_path

util.build_dirs(dir_name)
logger = util.setup_logger(name=args.exp_name, log_file= dir_name / Path("log_file.log"))
logger.info("PyTorch Version: %s" % (torch.__version__))
logger.info(f"Training generator of architecture : {model_name} ")
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
    logger.info("GPU List: %s" % (device_list))
else:
    device = torch.device('cpu')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if args.dataset == "ImageNet":
    mean = [0.4636, 0.4745, 0.4375]
    std = [0.2335, 0.2329, 0.2382]
elif args.dataset == "MNIST":
    mean = [0.1307, 0.1307, 0.1307]
    std = [0.3081, 0.3081, 0.3081]
elif args.dataset == "CINIC":
    mean = [0.47889522, 0.47227842, 0.43047404]
    std = [0.24205776, 0.23828046, 0.25874835]
else:
    mean = [x/255.0 for x in [125.3, 123.0, 113.9]]
    std = [x/255.0 for x in [63.0, 62.1, 66.7]]


normalize = transforms.Normalize(mean=mean,
                                     std=std)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    ])
transform_train = transform_test

if args.dataset == "ImageNet":
    logger.info(f"ImageNet will be used with a resolution, both for the generator and the learning model of {args.resolution}")
    train_dataset_full = datasets.ImageNet(root="/scratch/data_share_ChDi/ILSVRC2012", transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((args.resolution,args.resolution))]))
    targets = [x for x in range(250)]
    indices = [i for i, label in enumerate(train_dataset_full.targets) if label in targets]
    train_dataset = Subset(train_dataset_full, indices)
    dataloader_train = DataLoader(train_dataset, batch_size= args.b, shuffle = True, num_workers = 6)
    logger.info(f"Thus we have a length of {len(train_dataset)}")

    test_dataset_full = datasets.ImageNet(root="/scratch/data_share_ChDi/ILSVRC2012", transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((args.resolution,args.resolution))]), split = "val")
    targets_test = [x for x in range(250)]
    indices_test = [i for i, label in enumerate(test_dataset_full.targets) if label in targets_test]
    test_dataset = Subset(test_dataset_full, indices_test)
    dataloader_test = DataLoader(test_dataset, batch_size= args.b, shuffle = True, num_workers=6)
    logger.info(f"And a length of {len(test_dataset)} for testing/validation")
elif args.dataset == "CINIC":
    logger.info(f"Dataset used will be ImageNet Part of CINIC")
    train_dataset = datasets.ImageFolder("/ceph/csedu-scratch/project/dvlijmen/MSDS-tryout/datasets_CINIC/cinic-10-imagenet/train/", transform=transform_test)
    dataloader_train = DataLoader(train_dataset, batch_size= args.b, shuffle=True, num_workers = 16)
elif args.dataset == "MNIST":
    logger.info(f"dataset used will be MNIST")
    train_dataset = datasets.MNIST(root=dataset_path, train=True,
                                            download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1) )]))
    test_dataset = datasets.MNIST(root=dataset_path, train=False,
                                            download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1) )]))
    dataloader_train = DataLoader(train_dataset, batch_size= args.b, shuffle = True, num_workers = 2)
    dataloader_test = DataLoader(test_dataset, batch_size= args.b, shuffle = True, num_workers = 2)
else:
    train_dataset = datasets.CIFAR10(root=dataset_path, train=True,
                                            download=True, transform=transform_train)
    if args.Random_20 is not None:
        gen = torch.Generator()
        gen.manual_seed(args.Random_20)
        train_dataset, _ = random_split(train_dataset, [10000,40000], generator=gen)

    test_dataset =  datasets.CIFAR10(root=dataset_path, train=False,
                                             download=True, transform=transform_test)
    dataloader_train = DataLoader(train_dataset, batch_size= args.b, shuffle = True, num_workers = 2)
    dataloader_test = DataLoader(test_dataset, batch_size= args.b, shuffle = True, num_workers=2)

if args.discr_seed:
    # This seed is set for the reproducibility of the randomly initalized discriminator.
    setup_seed(args.discr_seed)
    logger.info(f"A seed has been set with the value {args.discr_seed} for the random init. of the discriminator")
else:
    if args.model == "Linear":
        discriminator = linearModel(train_dataset[0][0].size().numel(), 10)
        logger.info("Randomly initalized and fixed discriminator used is a Linear Model")
    else:
        if args.dataset == "ImageNet":
            discriminator = ResNet18(250)
        else:
            discriminator = ResNet18(10)
        logger.info("Randomly initalized and fixed discriminator used is a ResNet18")
      
generator = GeneratorResnet(data_dim='low')
if args.load_gen is not None:
    gen_path = args.load_gen
    gen_checkpoint = util.load_model(filename=gen_path, model=generator, optimizer= None, scheduler= None)
    generator.load_state_dict(gen_checkpoint['model_state_dict'])
    logger.info(F"Loaded a generator from {gen_path}")

if torch.cuda.is_available():
    generator = generator.cuda()   
    discriminator = discriminator.cuda() 

#Making sure that the discriminator cannot be learned
discriminator.eval()
for param in discriminator.parameters():
    param.requires_grad = False

epochs = args.epochs
optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta_1, 0.999))
criterion = torch.nn.CrossEntropyLoss()
logger.info(f"Optimizer used: {optimizer}")
original_loss = util.AverageMeter()
original_acc = util.AverageMeter()
original_logit_value = util.AverageMeter()

augment_resolution = train_dataset[0][0].size()[1]

# Below we determine the original accuracy of the fixed discriminator. Given that this network is randomly initialized, this could namely be different over runs
for images,labels in dataloader_train:
    images, labels = images.to(device), labels.to(device)

    # If noaugment is NOT true we require augmentation of the image before feeding it to the generator
    if not args.noaugment:
        aug_transforms = transforms.Compose([transforms.RandomCrop(augment_resolution, padding=4),
                        transforms.RandomHorizontalFlip()])
        images = transforms.Lambda(lambda ims: torch.stack([aug_transforms(image) for image in ims]))(images)

    disc_out = discriminator(normalize(images))
    correct_label_logit_value = torch.squeeze(torch.gather(torch.softmax(disc_out, dim = 1), 1, torch.unsqueeze(labels, dim = 1)))

    loss = criterion(disc_out, labels)

    mean_logit_value = torch.mean(correct_label_logit_value)

    original_l = criterion(disc_out,labels)
    original_a = util.accuracy(disc_out, labels, topk = (1,))[0]

    original_loss.update(original_l.item(), images.size(0)) 
    original_acc.update(original_a.item(), images.size(0))
    original_logit_value.update(mean_logit_value.item(), images.size(0))

orig_logit_val = original_logit_value.avg
original_loss_value = original_loss.avg
original_acc_value = original_acc.avg

logger.info(f"original loss value: {original_loss_value:.3f}")
logger.info(f"original acc: {original_acc_value:.3f}")
logger.info(f"original logit_value: {orig_logit_val:.3f}")

losses_train = [original_loss_value]
losses_difference = []
logit_values = [orig_logit_val]
# This value is set to havae a callback for the best loss value found. 8 seems to be around the lowest original loss of the discriminator we have seen
best_train = 8
acc_train = [original_acc_value]

generator.train() 
# This sets a seed before the training starts, to have reproducibilit of the transforms (augmentations) used during training.
if args.transform_seed is not None:
        setup_seed(args.transform_seed)

for epoch in tqdm(range(int(epochs)),desc='Training Epochs'):    
    generator.train()
    
    losses = util.AverageMeter()
    top1 = util.AverageMeter()
    logit_value = util.AverageMeter()
    mean_differences = []
    for images, labels in dataloader_train:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()        
        
        # See the comment within the determination of the original loss
        if not args.noaugment:
            aug_transforms = transforms.Compose([transforms.RandomCrop(augment_resolution, padding=4),
                            transforms.RandomHorizontalFlip()])
            images = transforms.Lambda(lambda ims: torch.stack([aug_transforms(image) for image in ims]))(images)     
        
        noise = generator(images).squeeze()
        ule = images + noise

        if not args.noclamp:
            ule = torch.clamp(ule, 0.0, 1.0)
        # This performs augmentation against the discriminator (Paper: Input Augmentation, thesis: Post Augmentation)
        if args.post_augment:
            post_aug_transform = transforms.Compose([transforms.RandomCrop(augment_resolution, padding=4),
                            transforms.RandomHorizontalFlip()])
            
            ule = transforms.Lambda(lambda ule: torch.stack([post_aug_transform(image) for image in ule]))(ule)

        noise_diffcheck = torch.clone(noise).detach().cpu().numpy()
        mean_differences.append(np.abs(np.mean(noise_diffcheck,axis = 0)))
    
        disc_out = discriminator(normalize(ule)) 
        
        correct_label_logit_value = torch.squeeze(torch.gather(torch.softmax(disc_out, dim = 1), 1, torch.unsqueeze(labels, dim = 1)))
        mean_logit_value = torch.mean(correct_label_logit_value)

        loss = criterion(disc_out, labels)
          
        prec1 = util.accuracy(disc_out, labels, topk = (1,))[0]
        
        loss.backward()
        optimizer.step()

        logit_value.update(mean_logit_value.item(), images.size(0))
        losses.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))

    mean_difference = np.mean(mean_differences, axis = 0)
    loss_difference = original_loss_value - losses.avg
    losses_difference.append(loss_difference)
    losses_train.append(losses.avg)
    logit_values.append(logit_value.avg)
    acc_train.append(top1.avg)   
    if top1.avg > best_train:
        best_train = top1.avg
        util.save_model(str(Path(dir_name) / Path(model_name + "best")), epoch, generator, optimizer, scheduler = None)
        logger.info(f"Model with training accuracy of {top1.avg:35f} saved at: " + str(dir_name) + "/"+ model_name + "best.pth")
    
    logger.info(f'Epoch {epoch+1:02}: | Mean correct logit value: {logit_value.avg:.5f} | Train Loss: {losses.avg:.5f} | Train Acc: {top1.avg:.3f} | Loss difference: {loss_difference:.3f} | Mean absolute noise value: max: {np.max(mean_difference):.3f}, min: {np.min(mean_difference):.3f}, mean: {np.mean(mean_difference):.3f}')

if args.saveLastModel:
    util.save_model(str(Path(dir_name) / Path(model_name + "last_epoch")), epochs, generator, optimizer, scheduler = None)
    logger.info("Final model saved at: " + str(dir_name) + "/"+ model_name + "last_epoch.pth")

if args.plot:
    util.plot_loss_acc(epochs, None, losses_train, None, acc_train, dir_name, model_name)
    logger.info("Plotted the results")

if args.saveLists:
    util.save_loss_acc(None, losses_train, None, acc_train, dir_name, model_name)
    logger.info("Saved the results")
