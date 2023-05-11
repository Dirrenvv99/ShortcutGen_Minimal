import random
from sched import scheduler
from matplotlib import pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision.models.mobilenet import MobileNetV2
# from torch.nn.functional import normalize
import numpy as np
# from sklearn.preprocessing import StandardScaler, normalize
import argparse
from tqdm import tqdm
from models.Linear import linearModel
from models.VGG import VGG11, VGG13, VGG16, VGG19
from models.CNN_detector import CNN_detector, LeNet, CNN
from models.wideresnet import WideResNet
from models.inception_resnet_v1 import InceptionResnetV1
from models.ResNet import ResNet18
from models.MLP import MLP
from models.generatorResNet import GeneratorResnet, GeneratorResnet_P_Ensemble, GeneratorResnetEnsemble
import util
from partialCusomDataset import More_GenPerturbedDataset, NoiseOverDifferentDataset, perturbDataset, generatorPerturbedDataset, GenPerturbedDataset, ClassShuffleDataset, two_GenPerturbedDataset

parser = argparse.ArgumentParser(description='Linear Model check')
parser.add_argument('--b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--dataset', type= str, default = "CIFAR10",
                    help="Which dataset to train on (Options: ImageNet, CIFAR10default: CIFAR10)")
parser.add_argument('--resolution', default = 32, type = int,
                    help = "Resolution if ImageNet is used (Default: 32)")
parser.add_argument('--gen_path', type =str, default = "./FixedDiscriminator/Only_AdvGanLoss/GenResNet_OnlyAdvGANhigh_train.pt",
                    help = "Path to the generator used to generate the perturbed images given a original image")
parser.add_argument('--gen_path_2', type =str, default = None,
                    help = "Path to the second generator used to generate the perturbed images given a original image")
parser.add_argument('--gen_path_3', type =str, default = None,
                    help = "Path to the third generator used to generate the perturbed images given a original image")
parser.add_argument('--gen_path_4', type =str, default = None,
                    help = "Path to the fourth generator used to generate the perturbed images given a original image")
parser.add_argument('--no_validation', type = bool, default = False, 
                    help = "False: validation is performed; True: validation is not performed (Note that it is assumed that test set can be recreated from train = False within Pytorch datasets")
parser.add_argument('--epochs', type = int, default = 50, 
                    help = "amount of epochs performed (defult = 50)")
parser.add_argument('--exp_name', type =str,
                    help = "String that will be used as name for the specific directory the results will be saved in (Note: This has NO default value)")
parser.add_argument('--testIter', default= 1 , type=int,
                    help='amount of iterations between validation on testset (default = 1)')
parser.add_argument('--ensemble_numbers', type =int, default = None, nargs="+",
                    help = "3 numbers that if given represent the ensemble of generators to use (e.g. 2 1 3 will give a enseemble of 2 postaug generator, 1 aug generator and 3 no aug generators). If defaulted to None, normal generator will be used")
parser.add_argument('--discr_seed', default= None , type=int,
                    help='Settting the seed for the discriminator/classifier. If set to None no seed will be set. (default = None)')
parser.add_argument('--transform_seed', type = int, default = None, 
                    help = "If not none, will be used to set the seed just before the start of the generation of the perturbed dataset. Setting the seed for the random augmentations that are used in presentation to the generator")
parser.add_argument('--random_20', type = int, default = None, 
                    help = "If not none, will be used to set the seed just before the start of the generation of the perturbed dataset. Setting the seed for the random augmentations that are used in presentation to the generator")
parser.add_argument('--dir_name', type =str, default = "MethodEvaluation",
                    help = "String that will be used as name for the directory that the results will be saved in; This is not equal to the exp_name")
parser.add_argument('--plot', type = bool, default = False, 
                    help = "If true plots the training and validation losses and accuracy over epochs")
parser.add_argument('--old_optimizer', type = bool, default = False, 
                    help = "If true old optimzer (as reported within ULE paper) is used")
parser.add_argument('--different_noise', type = str, default = None, 
                    help = "Path to directory of the noise and labels for DifferentNoise over dataset")
parser.add_argument('--saveLists', type = bool, default = False, 
                    help = "If true save the numpy arrays containing the loss and accuracy for training and validation")
parser.add_argument('--model', type =str, default = "RN18",
                    help = "Model that will be trained as discriminator (Options:  WideResNet, RN-18, CNN, VGG11, MLP, mobilenetv2 and GoogleLeNet; default: RN18)")
parser.add_argument('--noaugment', type = bool, default = False, 
                    help = "True: Augment the training data; False: No augmentation is used (default : False)")
parser.add_argument('--grayaug', type = bool, default = False, 
                    help = "True: Augment and grayscale the training data; False: No grayscale is used (default : False)")
parser.add_argument('--perturbed_out', type = bool, default = False, 
                    help = "Set to true if generator directly outputs the ule")
parser.add_argument('--grayaug_gen', type = bool, default = False, 
                    help = "True: noise of generator is grayscaled as in training; False: No grayscale is used (default : False)")
parser.add_argument('--gray_presented', type = bool, default = False, 
                    help = "True: pictures are first grayscaled before the generator sees them; False: This is not done (default : False)")
parser.add_argument('--aug_presented', type = bool, default = False, 
                    help = "True: pictures are first augemented (flip + crop) before the generator sees them; False: This is not done (default : False)")
parser.add_argument('--shuffle', type = bool, default = False, 
                    help = "True: shuffle the perturbations within the class to check whether class-wise is emulated; False: This shuffle is not performed. (default : False'; Note: for now do NOT use when augmentation/Grayscaling is also used!)")
parser.add_argument('--noise_only', type = bool, default = False, 
                    help = "True: Classifier will be learned on the noise only, and not on the perturbed image. Noise will still be generated against chosen dataset")
parser.add_argument('--saveLastModel', type = bool, default = False, 
                    help = "True: Last classifier will be saved (additionally); False: Last Classifier will not be saved (default: False)")
parser.add_argument('--SaveHighestModel', type = bool, default = False, 
                    help = "True: Classifier with highest accuracy will be saved; False: Opposite (default: False)")
parser.add_argument('--noclamp', type = bool, default = False, 
                    help = "True:  Clamping the values between 0 and 1 will not be performed (default: False)")
parser.add_argument('--only_additive', type = bool, default = False, 
                    help = "True:  Only noise pixel values with a postive value are used. As such only features that are tried to be added are not removed (default: False)")
parser.add_argument('--only_subtractive', type = bool, default = False, 
                    help = "True:  Only noise pixel values with a negative value are used. As such only features that are tried to be subtracted are not removed  from the noise (default: False)")
parser.add_argument('--comb_method', type = str, default = None, 
                    help = "Method of combination if two generators are used, 2 generators will not be used if this is set to None. (Options: Linear Combination, Max, Signed Max, Pixelchoice; Default: None)")
args = parser.parse_args()

dir_name = args.dir_name + "/" + args.exp_name
if args.model == "CNN":
    model_name = "CNN"
elif args.model == "WideResNet":
    model_name = "WideResNet"
elif "VGG" in args.model:
    model_name = "VGG11"
elif args.model == "MLP":
    model_name = "MLP"
elif args.model == "Linear":
    model_name = "Linear"
elif "Mobile" in args.model:
    model_name = "MobileNet_v2"
elif "GoogLe" in args.model or "Inception" in args.model:
    model_name = "GoogleLeNet"
else:
    model_name = "ResNet18"

dataset_path = "../datasets"

util.build_dirs(dir_name)
logger = util.setup_logger(name=args.exp_name, log_file= dir_name + "/log_file.log")
logger.info("PyTorch Version: %s" % (torch.__version__))
logger.info(f"Model to be used as discriminator: {model_name}, evaluation on dataset : {args.dataset}")
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
    logger.info("GPU List: %s" % (device_list))
else:
    device = torch.device('cpu')

num_of_classes = 10
if args.dataset == "ImageNet":
    num_of_classes = 100

if args.dataset == "ImageNet":
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
else:
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

if args.grayaug:
    logger.info("Grayscale augmentation is used")
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.Grayscale(3)])
elif args.noaugment:
    logger.info("No augmentation is used")
    transform_train = None
else:
    logger.info("Normal augmentation is used")
    transform_train = transforms.Compose([transforms.RandomCrop(args.resolution, padding=4),
                            transforms.RandomHorizontalFlip()])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    ])
if args.dataset == "ImageNet":
    mean = [0.4636, 0.4745, 0.4375]
    std = [0.2335, 0.2329, 0.2382]
elif args.dataset == "MNIST":
    mean = [0.1307, 0.1307, 0.1307]
    std = [0.3081, 0.3081, 0.3081]
else:
    mean = [x/255.0 for x in [125.3, 123.0, 113.9]]
    std = [x/255.0 for x in [63.0, 62.1, 66.7]]

                               
def normalize_apart(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
    return t

# different_noise means we are evaluating noise not generated by a generator, but saved somewhere. So here we make sure want to use a generator
if args.different_noise is None:
    if args.ensemble_numbers is not None:
        if args.ensemble_numbers[0] != 0 and args.ensemble_numbers[1] + args.ensemble_numbers[2] ==0:
            generator = GeneratorResnet_P_Ensemble(args.ensemble_numbers[0], resolution=args.resolution)
        else:
            if args.comb_method is not None: 
                generator = GeneratorResnetEnsemble(args.ensemble_numbers[0], args.ensemble_numbers[1], args.ensemble_numbers[2], maximum_combine=True)
            else:
                generator = GeneratorResnetEnsemble(args.ensemble_numbers[0], args.ensemble_numbers[1], args.ensemble_numbers[2], maximum_combine=False)
    else:
        generator = GeneratorResnet(data_dim='low')
    checkpoint_generator = util.load_model(args.gen_path, model = generator, optimizer= None, scheduler= None)
    generator.load_state_dict(checkpoint_generator['model_state_dict'])

    if torch.cuda.is_available():
        generator = generator.cuda()
    generator.eval()
    for param in generator.parameters():
        param.requires_grad = False
    logger.info("Generator used to generate the perturbed images downloaded from: " +  args.gen_path)

    if args.gen_path_2 is not None:
        logger.info(f"Comb Method is " + args.comb_method)
        generator_2 = GeneratorResnet(data_dim='low')
        checkpoint_generator_2 = util.load_model(args.gen_path_2, model = generator_2, optimizer= None, scheduler= None)
        generator_2.load_state_dict(checkpoint_generator_2['model_state_dict'])

        if torch.cuda.is_available():
            generator_2 = generator_2.cuda()
        generator_2.eval()
        logger.info("Second generator used to generate the perturbed images downloaded from: " +  args.gen_path_2)
        if args.gen_path_3 is not None:
            generator_3 = GeneratorResnet(data_dim='low')
            checkpoint_generator_3 = util.load_model(args.gen_path_3, model = generator_3, optimizer= None, scheduler= None)
            generator_3.load_state_dict(checkpoint_generator_3['model_state_dict'])

            if torch.cuda.is_available():
                generator_3 = generator_3.cuda()
            generator_3.eval()
            logger.info("Third generator used to generate the perturbed images downloaded from: " +  args.gen_path_3)
            gen_list = [generator, generator_2, generator_3]
            if args.gen_path_4 is not None:
                generator_4 = GeneratorResnet(data_dim='low')
                checkpoint_generator_4 = util.load_model(args.gen_path_4, model = generator_4, optimizer= None, scheduler= None)
                generator_4.load_state_dict(checkpoint_generator_4['model_state_dict'])

                if torch.cuda.is_available():
                    generator_4 = generator_4.cuda()
                generator_4.eval()
                logger.info("Fourth generator used to generate the perturbed images downloaded from: " +  args.gen_path_4)
                gen_list.append(generator_4)

epochs = args.epochs
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
if args.discr_seed is not None:
    setup_seed(args.discr_seed)
    if args.discr_seed == 31:
        model = ResNet18(num_of_classes)
    logger.info(f"Seed of the classifier/discriminator is set to {args.discr_seed}")
if args.model == "CNN":
    #TODO: Implement different optimizer ? (ADAM instead of SGD) -> See results from first eval
    model = CNN(num_of_classes)
    logger.info("Model used is a CNN")
elif args.model == "WideResNet":
    #TODO change to num_classes if needed
    model = WideResNet(28, 10, 10, 0)
    logger.info("Model used is a WideResNet")
elif "VGG" in args.model:
    model = VGG11(num_of_classes)
    logger.info("Model used is a VGG11")
elif args.model == "MLP":
    #TODO change to num_classes if needed
    model = MLP()
    logger.info("Model used is a MLP")
elif args.model == "Linear":
    model = linearModel(32*32*3,num_of_classes)
    logger.info("Model used is a linear layer")
elif "Mobile" in args.model:
    model = MobileNetV2(num_of_classes)
    logger.info("Model used is a MobileNetV2")
elif "GoogLe" in args.model or "Inception" in args.model:
    #TODO: Get this to work for CIFAR10 -> Do we want to? 
    model = InceptionResnetV1(classify= True, num_classes= 10)
    logger.info("Model used is a GoogLeNet/InceptionResNetV1")
else:
    model = ResNet18(num_of_classes)
    logger.info("Model used is a RN18")
if torch.cuda.is_available():
    model = model.cuda()

# Specific configuration is taken from Huang et Al.
# choice between different optimizers we used. Old optimizer is worse
if args.old_optimizer:
    optimizer = torch.optim.SGD(model.parameters(), 0.025, momentum= 0.9)
else:
    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                          momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= epochs) 

criterion = torch.nn.CrossEntropyLoss()

logger.info(f"Optimizer used for the classifier: {optimizer}")
logger.info(f"Scheduler used for the classifier: {scheduler}")

iter = 0
losses_test_l = []
losses_train = []

best_test = 90
best_train = 90

acc_train = []
acc_test = []

model.train() 

if args.dataset == "ImageNet":
    logger.info(f"ImageNet will be used with a resolution, both for the generator and the learning model of {args.resolution}")
    train_dataset_full = datasets.ImageNet(root="/scratch/data_share_ChDi/ILSVRC2012", transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((args.resolution,args.resolution))]))
    targets = [705,895,14,555,344,256,510,261,31,22]
    indices = [i for i, label in enumerate(train_dataset_full.targets) if label in targets]
    train_dataset_parent = Subset(train_dataset_full, indices)

    logger.info(f"Thus we have a length of {len(train_dataset_parent)}")

    test_dataset_full = datasets.ImageNet(root="/scratch/data_share_ChDi/ILSVRC2012", transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((args.resolution,args.resolution))]), split = "val")
    targets_test = [705,895,14,555,344,256,510,261,31,22]
    indices_test = [i for i, label in enumerate(test_dataset_full.targets) if label in targets_test]
    test_dataset = Subset(test_dataset_full, indices_test)
    dataloader_test = DataLoader(test_dataset, batch_size= args.b, shuffle = True, num_workers=4)
    logger.info(f"And a length of {len(test_dataset)} for testing/validation")
elif args.dataset == "MNIST":
    logger.info(f"dataset used will be MNIST")
    train_dataset_parent = datasets.MNIST(root=dataset_path, train=True,
                                            download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1) )]))
    test_dataset = datasets.MNIST(root=dataset_path, train=False,
                                            download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1) )]))
    dataloader_test = DataLoader(test_dataset, batch_size= args.b, shuffle = True, num_workers = 2)

else:
    if args.aug_presented:
        logger.info("Pictures are augmented before they are presented to the generator while making the perturbed dataset")
        transform_presented = transforms.Compose([transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()])
        if args.transform_seed is not None:
            setup_seed(args.transform_seed)
        train_dataset_parent = datasets.CIFAR10(root=dataset_path, train=True,
                                            download=True, transform=transform_presented)
    else:
        train_dataset_parent = datasets.CIFAR10(root=dataset_path, train=True,
                                                download=True, transform=transform_test)
        if args.random_20 is not None:
            gen = torch.Generator()
            gen.manual_seed(args.random_20)
            _, train_dataset_parent = random_split(train_dataset_parent, [10000,40000], generator=gen)
        
    test_dataset =  datasets.CIFAR10(root=dataset_path, train=False,
                                             download=True, transform=transform_test)
    dataloader_test = DataLoader(test_dataset, batch_size= args.b, shuffle = True, num_workers = 2)

if args.grayaug_gen:
    logger.info("Generator that has grayscaled noise is used")
if args.different_noise is not None:
    perturbations = np.load(args.different_noise + "/perturbations.npy")
    labels = np.load(args.different_noise + "/labels.npy")
    train_dataset = NoiseOverDifferentDataset(train_dataset_parent, perturbations, labels, transform=transform_train)
    dataloader_train = DataLoader(train_dataset, batch_size= args.b, shuffle = True, num_workers=4)
# If true the noise is shuffled over the images per class. Thus images get noise not oriignally meant for themself
elif args.shuffle:
    train_dataset = ClassShuffleDataset(train_dataset_parent, generator, transform = transform_train, noclamp = args.noclamp, gen_ga = args.grayaug_gen, gray_presented = args.gray_presented)
    dataloader_train = DataLoader(train_dataset, batch_size= args.b, shuffle = True)
    logger.info("The shuffled Dataset has been created!")
elif args.gen_path_2 is not None:
    if args.gen_path_3 is not None:
        train_dataset = More_GenPerturbedDataset(train_dataset_parent, gen_list, comb_method= args.comb_method, transform= transform_train, batch_size_generator= args.b, noclamp= args.noclamp, gen_ga = args.grayaug_gen, gray_presented = args.gray_presented,noise_only= args.noise_only)
        logger.info(f"The perturbed dataset using {len(gen_list)} generators has been created")
    else:
        train_dataset = two_GenPerturbedDataset(train_dataset_parent, generator, generator_2, comb_method= args.comb_method, transform= transform_train, batch_size_generator= args.b, noclamp= args.noclamp, gen_ga = args.grayaug_gen, gray_presented = args.gray_presented, noise_only= args.noise_only)
        logger.info("The perturbed dataset using two generators has been created")
    dataloader_train = DataLoader(train_dataset, batch_size= args.b, shuffle = True)
        
else:
    if args.only_additive:
        logger.info("Only the added features are used within evaluation")
    train_dataset = GenPerturbedDataset(train_dataset_parent, generator, transform = transform_train, batch_size_generator= args.b, noclamp= args.noclamp, gen_ga = args.grayaug_gen, gray_presented = args.gray_presented, additive_features_only= args.only_additive, subtractive_features_only= args.only_subtractive, noise_only= args.noise_only, perturbed_out = args.perturbed_out)
    if args.dataset == "ImageNet":
        dataloader_train = DataLoader(train_dataset, batch_size= args.b, shuffle = True, num_workers=4)
    else:
        dataloader_train = DataLoader(train_dataset, batch_size= args.b, shuffle = True, num_workers = 2)    
    logger.info("The perturbed dataset has been created")

if args.dataset == "ImageNet":
    remap = {x:i for i, x in enumerate(targets)}

for epoch in tqdm(range(int(epochs)),desc='Training Epochs'):
    model.train()
    losses = util.AverageMeter()
    top1 = util.AverageMeter()
    for images, labels in dataloader_train:
        #mapping labels for imagenet to 0,...,9
        if args.dataset == "ImageNet":
            labels = torch.from_numpy(np.array([remap[int(labels[i])] for i in range(labels.size(0))])).type(labels.type())
        
        labels = labels.to(device)
        images = images.to(device)    

        optimizer.zero_grad()
        logits = model(normalize_apart(images)).squeeze()

        loss = criterion(logits, labels)
        prec1 = util.accuracy(logits, labels, topk = (1,))[0]
        
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
    losses_train.append(losses.avg)
    acc_train.append(top1.avg)   
    if top1.avg > best_train and args.SaveHighestModel:
        best_train = top1.avg
        util.save_model(dir_name + "/"+ model_name + "high_train.pt", epoch, model, optimizer, scheduler = None)
        logger.info(f"Model with training accuracy of {top1.avg:.3f} saved at: " + dir_name + "/"+ model_name + "high_train.pt")
    if iter % args.testIter == 0 and args.no_validation == False:
        with torch.no_grad():
            model.eval()
            losses_test = util.AverageMeter()
            top1_test = util.AverageMeter()
            for X_test_batch, y_test_batch in dataloader_test:
                if args.dataset == "ImageNet":
                    y_test_batch = torch.from_numpy(np.array([remap[int(y_test_batch[i])] for i in range(y_test_batch.size(0))])).type(y_test_batch.type())
                X_test_batch_original, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)

                #Put below line in comments if you want the test set to be clean. 
                # X_test_batch = generator(X_test_batch_original)

                # X_test_batch = torch.min(torch.max(X_test_batch, X_test_batch_original - 8/255), X_test_batch_original + 8/255)
                # X_test_batch = torch.clamp(X_test_batch, 0.0, 1.0)

                y_test_pred = model(normalize_apart(X_test_batch_original)).squeeze()   

                test_loss = criterion(y_test_pred, y_test_batch)
                test_acc = util.accuracy(y_test_pred, y_test_batch, topk = (1,))[0]       
                losses_test.update(test_loss.item(), X_test_batch_original.size(0))
                top1_test.update(test_acc.item(), X_test_batch_original.size(0))
            logger.info(f'Epoch {epoch+0:02}: | Train Loss: {losses.avg:.5f} | Test Loss: {losses_test.avg:.5f} | Train Acc: {top1.avg:.3f}| Test Acc: {top1_test.avg:.3f}')                
            losses_test_l.append(losses_test.avg)
            acc_test.append(top1_test.avg) 
            if top1_test.avg > best_test and args.SaveHighestModel:
                best_test = top1_test.avg
                util.save_model(dir_name + "/"+ model_name + "high_test.pt", epoch, model, optimizer, scheduler = None)
                logger.info(f"Model with test accuracy of {top1_test.avg:.3f} and training accuracy of {top1.avg:.3f} saved at: " + dir_name + "/"+ model_name + "high_test.pt")
    else:
        logger.info(f'Epoch {epoch+1:02}: | Train Loss: {losses.avg:.5f} | Train Acc: {top1.avg:.3f}')
    if args.saveLastModel and iter == 5:
        util.save_model(dir_name + "/"+ model_name + "_victim_epoch_5", epochs, model, optimizer, scheduler = None)
        logger.info("Final model saved at: " + dir_name + "/"+ model_name + "_victim_epoch_5")
    iter += 1
    scheduler.step()

if args.saveLastModel:
    util.save_model(dir_name + "/"+ model_name + "_victim_last_epoch", epochs, model, optimizer, scheduler = None)
    logger.info("Final model saved at: " + dir_name + "/"+ model_name + "_victim_last_epoch")

if args.plot:
    util.plot_loss_acc(epochs, losses_test_l, losses_train, acc_test, acc_train, dir_name, model_name)
    logger.info("Plotted the results")

if args.saveLists:
    util.save_loss_acc(losses_test_l, losses_train, acc_test, acc_train, dir_name, model_name)
    logger.info("Saved the results")
