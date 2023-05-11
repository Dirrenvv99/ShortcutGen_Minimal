import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import util
from collections import Counter
from skimage.transform import resize



def signed_absolute_maximum(list):
    orig_shape = np.array(list[0]).shape
    list = np.array([i.flatten() for i in list])

    list = list[np.argmax(np.abs(list), axis=0), np.arange(list.shape[1])].reshape(orig_shape)

    return list

class partialCustomDataset(Dataset):
    '''custom dataset that contains both, with the correct labels'''
    def __init__(self, parent, noise, train = True):
        self.parent = parent
        self.noise = noise
        noise_length = noise.size(dim=0)
        self.indices = np.random.choice(noise_length,  noise_length//2, replace = False)

        if len(self.parent) != self.noise.size(dim=0):
            print("Note that there is a difference in size between the dataset and the noise tensor given")
            
    def __len__(self):
        return self.noise.size(dim=0)

    def __getitem__(self,idx):
        if idx in self.indices:
            return self.parent[idx][0], 0
        else:
            return self.parent[idx][0] + self.noise[idx], 1

class NoiseDataset(Dataset):
    def __init__(self, parent, noise_memory, transform = None):
        self.parent = parent
        batch_s = 64
        self.transform = transform
        image_shape = (parent[0][0].shape[-2], parent[0][0].shape[-1])
        new_noise = np.empty((len(parent),3,*image_shape))
        dataloader = DataLoader(parent, batch_s, shuffle = False)

        for i, (images, _) in tqdm(enumerate(dataloader)):
            if (i*batch_s + batch_s) <= len(parent):
                noise_batch = noise_memory[i * batch_s: (i*batch_s + batch_s)]
            else:
                noise_batch = noise_memory[i * batch_s:]
            per_images = noise_batch + images 
            per_images = torch.clamp(per_images, 0.0, 1.0)
            noise_per_batch = (per_images - images).detach().cpu().numpy()

            if (i*batch_s + batch_s) <= len(parent):
                new_noise[i * batch_s: (i*batch_s + batch_s)] = noise_per_batch
            else:
                new_noise[i * batch_s:] = noise_per_batch

        self.noise = torch.from_numpy(new_noise).type(torch.FloatTensor)

            
    def __len__(self):
        return self.noise.size(dim=0)

    def __getitem__(self,idx):
        image = self.parent[idx][0] + self.noise[idx]
        if self.transform is not None:
            image = self.transform(image)
        
        return image, self.parent[idx][1]

class ClassNoiseDataset(Dataset):
    def __init__(self, parent, noise_memory, transform = None):
        self.parent = parent
        self.transform = transform
        self.noise = noise_memory

            
    def __len__(self):
        return len(self.parent)

    def __getitem__(self,idx):
        image = self.parent[idx][0] + self.noise[self.parent[idx][1]]
        if self.transform is not None:
            image = self.transform(image)
        
        return image, self.parent[idx][1]

class NoiseOverDifferentDataset(Dataset):
    def __init__(self, parent, perturbations, labels, transform = None):
        self.parent = parent
        self.transform = transform
        image_shape = (parent[0][0].shape[-2], parent[0][0].shape[-1])
        noise = np.empty((len(parent), 3, *image_shape))
        # amount_of_classes = dict(Counter(parent.targets))
        targets_parent = np.array([parent[i][1] for i in range(len(parent))])
        remap = {x:i for i, x in enumerate(targets_parent)}
        targets_parent = np.array(remap[int(i)] for i in targets_parent)


        #it is assumed that all images within the parent dataset are of the same shape
        if (parent[0][0].shape[-3],parent[0][0].shape[-2], parent[0][0].shape[-1]) != perturbations[0].shape:
            new_shape = (parent[0][0].shape[-3],parent[0][0].shape[-2], parent[0][0].shape[-1])
            new_perturbations = np.empty((len(perturbations), *new_shape))
            for i, perturb in enumerate(perturbations):
                new_perturbations[i] = resize(perturb, new_shape)
            perturbations = new_perturbations
        
        for i in range(max(labels)):
            noise_with_label = perturbations[np.where(labels == i)[0]]
            idxs_parent = np.where(targets_parent == i)[0]
            if len(idxs_parent) < len(noise_with_label):
                noise[idxs_parent] = noise_with_label[:len(idxs_parent)]
            elif len(idxs_parent) == len(noise_with_label):
                noise[idxs_parent] = noise_with_label
            else:
                noise[idxs_parent] = np.resize(noise_with_label, (len(idxs_parent), *noise_with_label[0].shape))
        self.noise = torch.from_numpy(noise).type(torch.FloatTensor)

    def __len__(self):
        return len(self.parent)

    def __getitem__(self, idx):
        image = self.parent[idx][0] + self.noise[idx]
        if self.transform is not None:
            return self.transform(image), self.parent[idx][1]
        else:
            return image, self.parent[idx][1]

class GenPerturbedDataset(Dataset):
    def __init__(self, parent, generator, transform = None, noclamp = False, gen_ga = False, gray_presented = False, batch_size_generator = 128, additive_features_only = False, subtractive_features_only = False, noise_only = False, perturbed_out = False):
        self.parent = parent
        self.transform = transform
        batch_s = batch_size_generator
        dataloader = DataLoader(parent, batch_s, shuffle = False)
        image_shape = (parent[0][0].shape[-2], parent[0][0].shape[-2])
        noise = np.empty((len(parent),3,*image_shape))
        if torch.cuda.is_available():
            generator = generator.cuda()
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        generator.eval()
               

        for i, (images, _) in tqdm(enumerate(dataloader)):
            images_original = images.to(device)
            if gray_presented:
                images_original = transforms.Grayscale(3)(images_original)
            if perturbed_out:  # If the model already outputs fully perturbed image
                ule = generator(images_original)
                ule = torch.min(torch.max(ule, images_original - 8/255), images_original + 8/255)
                noise_output = ule - images_original
            else:  # If the model only outputs the noise
                noise_output = generator(images_original)
            if gen_ga:
                noise_output = transforms.Grayscale(3)(noise_output)
            per_images = images_original + noise_output

            if not noclamp:
                per_images = torch.clamp(per_images, 0.0, 1.0)

            noise_batch = (per_images - images_original).detach().cpu().numpy()


            if (i*batch_s + batch_s) <= len(parent):
                noise[i * batch_s: (i*batch_s + batch_s)] = noise_batch
            else:
                noise[i * batch_s:] = noise_batch
        
        if additive_features_only:
            noise = np.clip(noise, 0, None) 
        elif subtractive_features_only:
            noise = np.clip(noise, None, 0)
               
        self.noise = torch.from_numpy(noise).type(torch.FloatTensor)
        self.noise_only = noise_only

    def __len__(self):
        return len(self.parent)

    def __getitem__(self, idx):
        if self.noise_only:
            return self.noise[idx], self.parent[idx][1]
        image = self.parent[idx][0] + self.noise[idx]
        if self.transform is not None:
            image = self.transform(image)
        
        return image, self.parent[idx][1]

class two_GenPerturbedDataset(Dataset):
    def __init__(self, parent, generator, generator_2, comb_method = "Linear Combination", transform = None, noclamp = False, gen_ga = False, gray_presented = False, batch_size_generator = 128, noise_only = False):
        self.parent = parent
        self.transform = transform
        batch_s = batch_size_generator
        dataloader = DataLoader(parent, batch_s, shuffle = False)
        image_shape = (parent[0][0].shape[-2], parent[0][0].shape[-2])
        noise = np.empty((len(parent),3,*image_shape))
        if torch.cuda.is_available():
            generator = generator.cuda()
            generator_2 = generator_2.cuda()
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        generator.eval()
        generator_2.eval()    
        if comb_method == "Signed Max":
            print("Signed Max is used")           

        for i, (images, _) in tqdm(enumerate(dataloader)):
            images_original = images.to(device)
            if gray_presented:
                images_original = transforms.Grayscale(3)(images_original)
            noise_output = generator(images_original)
            noise_output_2 = generator_2(images_original)
            if gen_ga:
                noise_output = transforms.Grayscale(3)(noise_output)
                noise_output_2 = transforms.Grayscale(3)(noise_output_2)
            per_images = images_original + noise_output
            per_images_2 = images_original + noise_output_2

            if not noclamp:
                per_images = torch.clamp(per_images, 0.0, 1.0)
                per_images_2 = torch.clamp(per_images_2, 0.0, 1.0)

            noise_batch = (per_images - images_original).detach().cpu().numpy()
            noise_batch_2 = (per_images_2 - images_original).detach().cpu().numpy()

            #Combining the noise with method comb_method
            if comb_method == "Signed Max":
                noise_batch_full = np.array(util.signed_absolute_maximum([noise_batch, noise_batch_2]))
            elif comb_method == "Max":
                noise_batch_full = np.maximum(noise_batch, noise_batch_2)
            elif comb_method == "PixelChoice":
                print("Not yet implemented")
                noise_batch_full =  0.5 * noise_batch + 0.5 * noise_batch_2
            else:
                noise_batch_full =  0.5 * noise_batch + 0.5 * noise_batch_2

            if (i*batch_s + batch_s) <= len(parent):
                noise[i * batch_s: (i*batch_s + batch_s)] = noise_batch_full
            else:
                noise[i * batch_s:] = noise_batch_full
        
        self.noise = torch.from_numpy(noise).type(torch.FloatTensor)
        self.noise_only = noise_only
    def __len__(self):
        return len(self.parent)

    def __getitem__(self, idx):
        if self.noise_only:
            return self.noise[idx], self.parent[idx][1]
        else:
            image = self.parent[idx][0] + self.noise[idx]
            if self.transform is not None:
                image = self.transform(image)
            
            return image, self.parent[idx][1]


class More_GenPerturbedDataset(Dataset):
    def __init__(self, parent, gen_list, comb_method = "Linear Combination", transform = None, noclamp = False, gen_ga = False, gray_presented = False, batch_size_generator = 128, noise_only = False):
        self.parent = parent
        self.transform = transform
        batch_s = batch_size_generator
        dataloader = DataLoader(parent, batch_s, shuffle = False)
        image_shape = (parent[0][0].shape[-2], parent[0][0].shape[-2])
        noise = np.empty((len(parent),3,*image_shape))
        for index, _  in enumerate(gen_list):
            if torch.cuda.is_available():
                gen_list[index] = gen_list[index].cuda()
            gen_list[index].eval()
            for param in gen_list[index].parameters():
                param.requires_grad = False  
        if torch.cuda.is_available():  
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")            

        for i, (images, _) in tqdm(enumerate(dataloader)):
            images_original = images.to(device)
            if gray_presented:
                images_original = transforms.Grayscale(3)(images_original)
            noise_output = [generator(images_original) for generator in gen_list]
            if gen_ga:
                noise_output = [transforms.Grayscale(3)(output) for output in noise_output]
            per_images = [images_original + output for output in noise_output]

            if not noclamp:
                per_images = [torch.clamp(per_image, 0.0, 1.0) for per_image in per_images]

            noise_batch = [(per_image - images_original).detach().cpu().numpy() for per_image in per_images]

            #Combining the noise with method comb_method
            if comb_method == "Signed Max":
                noise_batch_full = np.array(util.signed_absolute_maximum(noise_batch))
            elif comb_method == "Max":
                noise_batch_full = np.max(noise_batch, axis = 0)
            elif comb_method == "PixelChoice":
                print("Not yet implemented")
                noise_batch_full = np.mean(noise_batch , axis = 0)
            else:
                noise_batch_full = np.mean(noise_batch , axis = 0)

            if (i*batch_s + batch_s) <= len(parent):
                noise[i * batch_s: (i*batch_s + batch_s)] = noise_batch_full
            else:
                noise[i * batch_s:] = noise_batch_full
        
        self.noise = torch.from_numpy(noise).type(torch.FloatTensor)
        self.noise_only = noise_only

    def __len__(self):
        return len(self.parent)

    def __getitem__(self, idx):
        if self.noise_only:
            return self.noise[idx], self.parent[idx][1]
        else:
            image = self.parent[idx][0] + self.noise[idx]
            if self.transform is not None:
                image = self.transform(image)
            
            return image, self.parent[idx][1]

class ClassShuffleDataset(Dataset):
    def __init__(self, parent, generator, noclamp = False, transform = None, gen_ga = False, gray_presented = False, batch_size_generator = 128):
        self.parent = parent
        self.transform = transform       
        noise_full = np.empty((len(parent),3,32,32))
        labels_full = np.empty((len(parent)))
        batch_s = batch_size_generator
        #need a dataloader that does not shuffle
        dataloader_create_noise = DataLoader(parent, batch_size= batch_s, shuffle = False)
        if torch.cuda.is_available():
            generator = generator.cuda()
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        generator.eval()

        for i, (images, labels) in tqdm(enumerate(dataloader_create_noise)):
            images_original = images.to(device)
            if gray_presented:
                images_original = transforms.Grayscale(3)(images_original)
            noise_output = generator(images_original)
            per_images = images_original + noise_output

            if not noclamp:
                per_images = torch.clamp(per_images, 0.0, 1.0)

            noise_batch = (per_images - images_original).detach().cpu().numpy()
            labels_batch = labels.detach().numpy()


            if (i*batch_s + batch_s) <= len(parent):
                noise_full[i * batch_s: (i*batch_s + batch_s)] = noise_batch
                labels_full[i * batch_s: (i*batch_s + batch_s)] = labels_batch
            else:
                noise_full[i * batch_s:] = noise_batch
                labels_full[i * batch_s:] = labels_batch
        num_of_classes = int(np.max(labels_full) + 1)
        new_noise = np.empty_like(noise_full)
        for i in range(num_of_classes):
            indices = (labels_full == i).nonzero()[0]
            if len(indices) > 1:
                new_noise[indices] = noise_full[indices[torch.randperm(indices.size)]]
            else:
                new_noise[indices] = noise_full[indices]
        self.noise = torch.from_numpy(new_noise).type(torch.FloatTensor)
    
    def __len__(self):
        return len(self.parent)

    def __getitem__(self,idx):
        image = self.parent[idx][0] + self.noise[idx]
        if self.transform is not None:
            image = self.transform(image)
        return  image, self.parent[idx][1]

class ShuffledDataset(Dataset):
    '''custom dataset that contains the labels that should be targeted instead of ground truth labels'''
    def __init__(self, parent):
        self.parent = parent
        labels = np.zeros(len(self.parent))
        all_labels = [self.parent[ix][1] for ix in range(len(self.parent))]
        max_label = np.max(all_labels)
        for i in range(len(self.parent)):
            ground_truth = self.parent[i][1]
            possibilities = np.append(np.arange(0,ground_truth), np.arange(ground_truth + 1, max_label + 1))
            random_label = np.random.choice(possibilities, 1)[0]
            labels[i] = random_label
        print("shuffled labels created")
        self.labels = torch.from_numpy(labels)
            
    def __len__(self):
        return len(self.parent)

    def __getitem__(self,idx):
        return self.parent[idx][0], self.labels[idx].type(torch.long)

class perturbDataset(Dataset):
    '''custom dataset that includes the generated noise, with the label corresponding to the CIFAR10 dataset'''
    def __init__(self, parent, noise, train = True):
        self.parent = parent
        self.noise = noise

        if len(self.parent) != self.noise.size(dim=0):
            print("Note that there is a difference in size between the dataset and the noise tensor given")
            
    def __len__(self):
        return self.noise.size(dim=0)

    def __getitem__(self,idx):
        orig = self.parent
        return orig[idx][0] + self.noise[idx], orig[idx][1]

class generatorPerturbedDataset(Dataset):
    '''Dataset that creates a (for now fully) perturbed dataset through usage of a given Generator, most likely not neccesary'''
    def __init__(self, parent, generator, train = True):
        self.parent = parent
        self.generator = generator

        self.generator.eval()
            
    def __len__(self):
        return len(self.parent)

    def __getitem__(self,idx):
        #Unsqueeze to generate fake batch size
        orig_image = self.parent[idx][0].unsqueeze(0)
        orig_label = self.parent[idx][1]

        if next(self.generator.parameters()).is_cuda:
            output = self.generator(orig_image).squeeze().cpu()
        return self.generator(orig_image).squeeze(), orig_label

class GenDetectionDataset(Dataset):
    '''Custom dataset to be able to check detection posibily of generative ULEs'''

    def __init__(self, parent, generator, batch_size_generator = 128):
        self.parent = parent
        self.generator = generator
        batch_s = batch_size_generator
        dataloader = DataLoader(parent, batch_s, shuffle = False)
        image_shape = (parent[0][0].shape[-1], parent[0][0].shape[-2])
        noise = np.empty((len(parent),3,*image_shape))
        if torch.cuda.is_available():
            generator = generator.cuda()
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        generator.eval()

        for i, (images, _) in tqdm(enumerate(dataloader)):
            images_original = images.to(device)
            noise_output = generator(images_original)
            per_images = images_original + noise_output
            per_images = torch.clamp(per_images, 0.0, 1.0)
            noise_batch = (per_images - images_original).detach().cpu().numpy()
            if (i*batch_s + batch_s) <= len(parent):
                noise[i * batch_s: (i*batch_s + batch_s)] = noise_batch
            else:
                noise[i * batch_s:] = noise_batch
        self.indices = np.random.permutation(len(parent))[:int((len(parent)//2))]
        self.noise = torch.from_numpy(noise).type(torch.FloatTensor)

    def __len__(self):
        return len(self.parent)

    def __getitem__(self, idx):
        if idx in self.indices:
            return self.parent[idx][0] + self.noise[idx], 1
        else:
            return self.parent[idx][0], 0


# class labelNoiseDataset(Dataset):
#     '''custom dataset that contains both, with the correct labels'''
#     def __init__(self, parent, noise, labels):
#         self.parent = parent
#         self.noise = noise
#         self.labels = labels

#         if len(self.parent) != self.noise.size(dim=0):
#             print("Note that there is a difference in size between the dataset and the noise tensor given")
            
#     def __len__(self):
#         return self.noise.size(dim=0)

#     def __getitem__(self,idx):
#         label = self.parent[idx][1]
#         if label in labels


