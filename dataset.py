import os
from random import randint

import cv2
import nibabel
import nibabel as nib
import numpy as np
import torch
from matplotlib import animation
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import imageio
import os
from PIL import Image
from torchvision import transforms
import pandas as pd
import random
from scipy import ndimage
import mat73
# helper function to make getting another batch of data easier


# from diffusion_training import output_img


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def make_pngs_anogan():
    dir = {
        "Train":     "./DATASETS/Train", "Test": "./DATASETS/Test",
        "Anomalous": "./DATASETS/CancerousDataset"
        }
    slices = {
        "00000": range(165, 205), "00002": range(177, 213), "00003": range(160, 190), "00005": range(160, 212),
        "00006": range(140, 200), "00008": range(135, 190), "00009": range(150, 205), "00011": range(130, 190),
        "00012": range(120, 180), "00014": range(170, 194), "00016": range(158, 195), "00017": range(155, 195),
        "00018": range(184, 213), "00019": range(158, 209), "00020": range(158, 210), "00021": range(164, 200),
        "00022": range(142, 200), "00024": range(160, 200), "00025": range(147, 210), "00026": range(155, 200)
        }
    center_crop = 235
    import os
    try:
        os.makedirs("./DATASETS/AnoGAN")
    except OSError:
        pass
    # for d_set in ["Train", "Test"]:
    #     try:
    #         os.makedirs(f"./DATASETS/AnoGAN/{d_set}")
    #     except OSError:
    #         pass
    #
    #     files = os.listdir(dir[d_set])
    #
    #     for volume_name in files:
    #         try:
    #             volume = np.load(f"{dir[d_set]}/{volume_name}/{volume_name}.npy")
    #         except (FileNotFoundError, NotADirectoryError) as e:
    #             continue
    #         for slice_idx in range(40, 120):
    #             image = volume[:, slice_idx:slice_idx + 1, :].reshape(256, 192).astype(np.float32)
    #             image = (image * 255).astype(np.int32)
    #             empty_image = np.zeros((256, center_crop))
    #             empty_image[:, 21:213] = image
    #             image = empty_image
    #             center = (image.shape[0] / 2, image.shape[1] / 2)
    #
    #             x = center[1] - center_crop / 2
    #             y = center[0] - center_crop / 2
    #             image = image[int(y):int(y + center_crop), int(x):int(x + center_crop)]
    #             image = cv2.resize(image, (64, 64))
    #             cv2.imwrite(f"./DATASETS/AnoGAN/{d_set}/{volume_name}-slice={slice_idx}.png", image)

    try:
        os.makedirs(f"./DATASETS/AnoGAN/Anomalous")
    except OSError:
        pass
    try:
        os.makedirs(f"./DATASETS/AnoGAN/Anomalous-mask")
    except OSError:
        pass
    files = os.listdir(f"{dir['Anomalous']}/raw_cleaned")
    center_crop = (175, 240)
    for volume_name in files:
        try:
            volume = np.load(f"{dir['Anomalous']}/raw_cleaned/{volume_name}")
            volume_mask = np.load(f"{dir['Anomalous']}/mask_cleaned/{volume_name}")
        except (FileNotFoundError, NotADirectoryError) as e:
            continue
        temp_range = slices[volume_name[:-4]]
        for slice_idx in np.linspace(temp_range.start + 5, temp_range.stop - 5, 4).astype(np.uint16):
            image = volume[slice_idx, ...].reshape(volume.shape[1], volume.shape[2]).astype(np.float32)
            image = (image * 255).astype(np.int32)
            empty_image = np.zeros((max(volume.shape[1], center_crop[0]), max(volume.shape[2], center_crop[1])))

            empty_image[9:165, :] = image
            image = empty_image
            center = (image.shape[0] / 2, image.shape[1] / 2)

            x = center[1] - center_crop[1] / 2
            y = center[0] - center_crop[0] / 2
            image = image[int(y):int(y + center_crop[0]), int(x):int(x + center_crop[1])]
            image = cv2.resize(image, (64, 64))
            cv2.imwrite(f"./DATASETS/AnoGAN/Anomalous/{volume_name}-slice={slice_idx}.png", image)

            image = volume_mask[slice_idx, ...].reshape(volume.shape[1], volume.shape[2]).astype(np.float32)
            image = (image * 255).astype(np.int32)
            empty_image = np.zeros((max(volume.shape[1], center_crop[0]), max(volume.shape[2], center_crop[1])))

            empty_image[9:165, :] = image
            image = empty_image
            center = (image.shape[0] / 2, image.shape[1] / 2)

            x = center[1] - center_crop[1] / 2
            y = center[0] - center_crop[0] / 2
            image = image[int(y):int(y + center_crop[0]), int(x):int(x + center_crop[1])]
            image = cv2.resize(image, (64, 64))
            cv2.imwrite(f"./DATASETS/AnoGAN/Anomalous-mask/{volume_name}-slice={slice_idx}.png", image)




def main(save_videos=True, bias_corrected=False, verbose=0):
    DATASET = "./DATASETS/CancerousDataset"
    patients = os.listdir(DATASET)
    for i in [f"{DATASET}/Anomalous-T1/raw_new", f"{DATASET}/Anomalous-T1/mask_new"]:
        try:
            os.makedirs(i)
        except OSError:
            pass
    if save_videos:
        for i in [f"{DATASET}/Anomalous-T1/raw_new/videos", f"{DATASET}/Anomalous-T1/mask_new/videos"]:
            try:
                os.makedirs(i)
            except OSError:
                pass

    for patient in patients:
        try:
            patient_data = os.listdir(f"{DATASET}/{patient}")
        except:
            if verbose:
                print(f"{DATASET}/{patient} Not a directory")
            continue
        for data_folder in patient_data:
            if "COR_3D" in data_folder:
                try:
                    T1_files = os.listdir(f"{DATASET}/{patient}/{data_folder}")
                except:
                    if verbose:
                        print(f"{patient}/{data_folder} not a directory")
                    continue
                try:
                    mask_dir = os.listdir(f"{DATASET}/{patient}/tissue_classes")
                    for file in mask_dir:
                        if file.startswith("cleaned") and file.endswith(".nii"):
                            mask_file = file
                except:
                    if verbose:
                        print(f"{DATASET}/{patient}/tissue_classes dir not found")
                    return
                for t1 in T1_files:
                    if bias_corrected:
                        check = t1.endswith("corrected.nii")
                    else:
                        check = t1.startswith("anon")
                    if check and t1.endswith(".nii"):
                        # try:
                        # use slice 35-55
                        img = nib.load(f"{DATASET}/{patient}/{data_folder}/{t1}")
                        mask = nib.load(f"{DATASET}/{patient}/tissue_classes/{mask_file}").get_fdata()
                        image = img.get_fdata()
                        if verbose:
                            print(image.shape)
                        if bias_corrected:
                            # image.shape = (256, 156, 256)
                            image = np.rot90(image, 3, (0, 2))
                            image = np.flip(image, 1)
                            # image.shape = (256, 156, 256)
                        else:
                            image = np.transpose(image, (1, 2, 0))
                        mask = np.transpose(mask, (1, 2, 0))
                        if verbose:
                            print(image.shape)
                        image_mean = np.mean(image)
                        image_std = np.std(image)
                        img_range = (image_mean - 1 * image_std, image_mean + 2 * image_std)
                        image = np.clip(image, img_range[0], img_range[1])
                        image = image / (img_range[1] - img_range[0])

                        np.save(
                                f"{DATASET}/Anomalous-T1/raw_new/{patient}.npy", image.astype(
                                        np.float32
                                        )
                                )
                        np.save(
                                f"{DATASET}/Anomalous-T1/mask_new/{patient}.npy", mask.astype(
                                        np.float32
                                        )
                                )
                        if verbose:
                            print(f"Saved {DATASET}/Anomalous-T1/mask/{patient}.npy")

                        if save_videos:
                            fig = plt.figure()
                            ims = []
                            for i in range(image.shape[0]):
                                tempImg = image[i:i + 1, :, :]
                                im = plt.imshow(
                                        tempImg.reshape(image.shape[1], image.shape[2]), cmap='gray', animated=True
                                        )
                                ims.append([im])

                            ani = animation.ArtistAnimation(
                                    fig, ims, interval=50, blit=True,
                                    repeat_delay=1000
                                    )

                            ani.save(f"{DATASET}/Anomalous-T1/raw_new/videos/{patient}.mp4")
                            if verbose:
                                print(f"Saved {DATASET}/Anomalous-T1/raw/videos/{patient}.mp4")
                            fig = plt.figure()
                            ims = []
                            for i in range(mask.shape[0]):
                                tempImg = mask[i:i + 1, :, :]
                                im = plt.imshow(
                                        tempImg.reshape(mask.shape[1], mask.shape[2]), cmap='gray', animated=True
                                        )
                                ims.append([im])

                            ani = animation.ArtistAnimation(
                                    fig, ims, interval=50, blit=True,
                                    repeat_delay=1000
                                    )

                            ani.save(f"{DATASET}/Anomalous-T1/mask_new/videos/{patient}.mp4")
                            if verbose:
                                print(mask.shape)
                                print(f"Saved {DATASET}/Anomalous-T1/raw/videos/{patient}.mp4")


def checkDataSet():
    resized = False
    mri_dataset = AnomalousMRIDataset(
            "DATASETS/CancerousDataset/EdinburghDataset/Anomalous-T1/raw", img_size=(256, 256),
            slice_selection="iterateUnknown", resized=resized
            # slice_selection="random"
            )

    dataset_loader = cycle(
            torch.utils.data.DataLoader(
                    mri_dataset,
                    batch_size=22, shuffle=True,
                    num_workers=2, drop_last=True
                    )
            )

    new = next(dataset_loader)

    image = new["image"]

    print(image.shape)
    from helpers import gridify_output
    print("Making Video for resized =", resized)
    fig = plt.figure()
    ims = []
    for i in range(0, image.shape[1], 2):
        tempImg = image[:, i, ...].reshape(image.shape[0], 1, image.shape[2], image.shape[3])
        im = plt.imshow(
                gridify_output(tempImg, 5), cmap='gray',
                animated=True
                )
        ims.append([im])

    ani = animation.ArtistAnimation(
            fig, ims, interval=50, blit=True,
            repeat_delay=1000
            )

    ani.save(f"./CancerousDataset/EdinburghDataset/Anomalous-T1/video-resized={resized}.gif")


def output_videos_for_dataset():
    folders = os.listdir("/Users/jules/Downloads/19085/")
    folders.sort()
    print(f"Folders: {folders}")
    for folder in folders:
        try:
            files_folder = os.listdir("/Users/jules/Downloads/19085/" + folder)
        except:
            print(f"{folder} not a folder")
            exit()

        for file in files_folder:
            try:
                if file[-4:] == ".nii":
                    # try:
                    # use slice 35-55
                    img = nib.load("/Users/jules/Downloads/19085/" + folder + "/" + file)
                    image = img.get_fdata()
                    image = np.rot90(image, 3, (0, 2))
                    print(f"{folder}/{file} has shape {image.shape}")
                    outputImg = np.zeros((256, 256, 310))
                    for i in range(image.shape[1]):
                        tempImg = image[:, i:i + 1, :].reshape(image.shape[0], image.shape[2])
                        img_sm = cv2.resize(tempImg, (310, 256), interpolation=cv2.INTER_CUBIC)
                        outputImg[i, :, :] = img_sm

                    image = outputImg
                    print(f"{folder}/{file} has shape {image.shape}")
                    fig = plt.figure()
                    ims = []
                    for i in range(image.shape[0]):
                        tempImg = image[i:i + 1, :, :]
                        im = plt.imshow(tempImg.reshape(image.shape[1], image.shape[2]), cmap='gray', animated=True)
                        ims.append([im])

                    ani = animation.ArtistAnimation(
                            fig, ims, interval=50, blit=True,
                            repeat_delay=1000
                            )

                    ani.save("/Users/jules/Downloads/19085/" + folder + "/" + file + ".mp4")
                    plt.close(fig)

            except:
                print(
                        f"--------------------------------------{folder}/{file} FAILED TO SAVE VIDEO ------------------------------------------------"
                        )



def load_datasets_for_test():
    args = {'img_size': (256, 256), 'random_slice': True, 'Batch_Size': 20, "dataset": "mri"}
    training, testing = init_datasets("./", args)

    ano_dataset = AnomalousMRIDataset(
            ROOT_DIR=f"data/brats/training", img_size=args['img_size'],
            slice_selection="random", resized=False
            )

    train_loader = init_dataset_loader(training, args)
    ano_loader = init_dataset_loader(ano_dataset, args)

    for i in range(5):
        new = next(train_loader)
        new_ano = next(ano_loader)
        output = torch.cat((new["image"][:10], new_ano["image"][:10]))
        plt.imshow(helpers.gridify_output(output, 5), cmap='gray')
        plt.show()
        plt.pause(0.0001)

def init_datasets(ROOT_DIR, args):
    if args["dataset"] == 'mri':
                
        training_dataset = MRIDataset(
                ROOT_DIR=f'{ROOT_DIR}DATASETS/Train/', img_size=args['img_size'], random_slice=args['random_slice']
                )
        testing_dataset = MRIDataset(
                ROOT_DIR=f'{ROOT_DIR}DATASETS/Test/', img_size=args['img_size'], random_slice=args['random_slice']
                )
    elif args["dataset"] == 'x-ray':
        training_dataset = TUMOR(input_size = args['img_size'], is_train=True, data_len=None)
        testing_dataset = TUMOR(input_size = args['img_size'], is_train=False, data_len=None)
    elif args["dataset"] == 'mura':
        training_dataset = MURA(input_size = args['image_size'], is_train=True, data_len=None, subset= "train")
        testing_dataset = MURA(input_size = args['image_size'], is_train=False, data_len=None, subset= "train")
    elif args["dataset"] == 'brats':
        training_dataset = BRATSDataset("./data/brats/training", test_flag=False)
        testing_dataset = BRATSDataset("./data/brats/testing", test_flag=True)
    elif args["dataset"] == 'brats2021':
        training_dataset = HnABRATS(ROOT_DIR, is_train=True)
        testing_dataset = HnABRATS(ROOT_DIR, is_train=False)
    elif args["dataset"] == 'pneumonia':
        training_dataset = Pneumonia(is_train=True)
        testing_dataset = Pneumonia(root = r"C:\Users\Admin\Dropbox\PC\Documents\Anomaly Detection\AnoDDPM\DATASETS\CheNemonia\chest_xray\train\PNEUMONIA", is_train=True)       
    else:
        print("No dataset found!")
    return training_dataset, testing_dataset


def init_dataset_loader(mri_dataset, args, shuffle=True):
    dataset_loader = cycle(
            torch.utils.data.DataLoader(
                    mri_dataset,
                    batch_size=args['Batch_Size'], shuffle=shuffle)
            )

    return iter(dataset_loader)


class DAGM(Dataset):
    def __init__(self, dir, anomalous=False, img_size=(256, 256), rgb=False, random_crop=True):
        # dir = './DATASETS/Carpet/Class1'
        if anomalous and dir[-4:] != "_def":
            dir += "_def"
        self.ROOT_DIR = dir
        self.anomalous = anomalous
        if rgb:
            norm_const = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        else:
            norm_const = ((0.5), (0.5))

        if random_crop:
            self.transform = transforms.Compose(
                    [
                        transforms.ToPILImage(),
                        transforms.ToTensor(),
                        transforms.Normalize(*norm_const)
                        ]
                    )
        else:
            self.transform = transforms.Compose(
                    [
                        transforms.ToPILImage(),
                        transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
                        transforms.ToTensor(),
                        transforms.Normalize(*norm_const)
                        ]
                    )
        self.rgb = rgb
        self.img_size = img_size
        self.random_crop = random_crop
        if anomalous:
            self.coord_info = self.load_coordinates(os.path.join(self.ROOT_DIR, "labels.txt"))
        self.filenames = os.listdir(self.ROOT_DIR)
        for i in self.filenames[:]:
            if not i.endswith(".png"):
                self.filenames.remove(i)
        self.filenames = sorted(self.filenames, key=lambda x: int(x[:-4]))

    def load_coordinates(self, path_to_coor):
        '''
        '''

        coord_dict_all = {}
        with open(path_to_coor) as f:
            coordinates = f.read().split('\n')
            for coord in coordinates:
                # print(len(coord.split('\t')))
                if len(coord.split('\t')) == 6:
                    coord_dict = {}
                    coord_split = coord.split('\t')
                    # print(coord_split)
                    # print('\n')
                    coord_dict['major_axis'] = round(float(coord_split[1]))
                    coord_dict['minor_axis'] = round(float(coord_split[2]))
                    coord_dict['angle'] = float(coord_split[3])
                    coord_dict['x'] = round(float(coord_split[4]))
                    coord_dict['y'] = round(float(coord_split[5]))
                    index = int(coord_split[0]) - 1
                    coord_dict_all[index] = coord_dict

        return coord_dict_all

    def make_mask(self, idx, img):
        mask = np.zeros_like(img)
        mask = cv2.ellipse(
                mask,
                (int(self.coord_info[idx]['x']), int(self.coord_info[idx]['y'])),
                (int(self.coord_info[idx]['major_axis']), int(self.coord_info[idx]['minor_axis'])),
                (self.coord_info[idx]['angle'] / 4.7) * 270,
                0,
                360,
                (255, 255, 255),
                -1
                )

        mask[mask > 0] = 255
        return mask

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        # print(repr(idx))
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {"filenames": self.filenames[idx]}
        if self.rgb:
            sample["image"] = cv2.imread(os.path.join(self.ROOT_DIR, self.filenames[idx]), 1)
            # sample["image"] = Image.open(os.path.join(self.ROOT_DIR, self.filenames[idx]), "r")
        else:
            sample["image"] = cv2.imread(os.path.join(self.ROOT_DIR, self.filenames[idx]), 0)

        if self.anomalous:
            sample["mask"] = self.make_mask(int(self.filenames[idx][:-4]) - 1, sample["image"])
        if self.random_crop:
            x1 = randint(0, sample["image"].shape[-1] - self.img_size[1])
            y1 = randint(0, sample["image"].shape[-2] - self.img_size[0])
            if self.anomalous:
                sample["mask"] = sample["mask"][x1:x1 + self.img_size[1], y1:y1 + self.img_size[0]]
            sample["image"] = sample["image"][x1:x1 + self.img_size[1], y1:y1 + self.img_size[0]]

        if self.transform:
            image = self.transform(sample["image"])
            if self.anomalous:
                sample["mask"] = self.transform(sample["mask"])
                sample["mask"] = (sample["mask"] > 0).float()
        sample["image"] = image.reshape(1, *self.img_size)

        return sample


class MVTec(Dataset):
    def __init__(self, dir, anomalous=False, img_size=(256, 256), rgb=True, random_crop=True, include_good=False):
        # dir = './DATASETS/leather'

        self.ROOT_DIR = dir
        self.anomalous = anomalous
        if not anomalous:
            self.ROOT_DIR += "/train/good"

        transforms_list = [transforms.ToPILImage()]

        if rgb:
            channels = 3
        else:
            channels = 1
            transforms_list.append(transforms.Grayscale(num_output_channels=channels))
        transforms_mask_list = [transforms.ToPILImage(), transforms.Grayscale(num_output_channels=channels)]
        if not random_crop:
            transforms_list.append(transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR))
            transforms_mask_list.append(transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR))
        transforms_list.append(transforms.ToTensor())
        transforms_mask_list.append(transforms.ToTensor())
        if rgb:
            transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        else:
            transforms_list.append(transforms.Normalize((0.5), (0.5)))
        transforms_mask_list.append(transforms.Normalize((0.5), (0.5)))
        self.transform = transforms.Compose(transforms_list)
        self.transform_mask = transforms.Compose(transforms_mask_list)

        self.rgb = rgb
        self.img_size = img_size
        self.random_crop = random_crop
        self.classes = ["color", "cut", "fold", "glue", "poke"]
        if include_good:
            self.classes.append("good")
        if anomalous:
            self.filenames = [f"{self.ROOT_DIR}/test/{i}/{x}" for i in self.classes for x in
                              os.listdir(self.ROOT_DIR + f"/test/{i}")]

        else:
            self.filenames = [f"{self.ROOT_DIR}/{i}" for i in os.listdir(self.ROOT_DIR)]

        for i in self.filenames[:]:
            if not i.endswith(".png"):
                self.filenames.remove(i)
        self.filenames = sorted(self.filenames, key=lambda x: int(x[-7:-4]))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # print(repr(idx))
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {"filenames": self.filenames[idx]}
        if self.rgb:
            sample["image"] = cv2.cvtColor(cv2.imread(os.path.join(self.filenames[idx]), 1), cv2.COLOR_BGR2RGB)
            # sample["image"] = Image.open(os.path.join(self.ROOT_DIR, self.filenames[idx]), "r")
        else:
            sample["image"] = cv2.imread(os.path.join(self.filenames[idx]), 0)
            sample["image"] = sample["image"].reshape(*sample["image"].shape, 1)

        if self.anomalous:
            file = self.filenames[idx].split("/")
            if file[-2] == "good":
                sample["mask"] = np.zeros((sample["image"].shape[0], sample["image"].shape[1], 1)).astype(np.uint8)
            else:
                sample["mask"] = cv2.imread(
                        os.path.join(self.ROOT_DIR, "ground_truth", file[-2], file[-1][:-4] + "_mask.png"), 0
                        )
        if self.random_crop:
            x1 = randint(0, sample["image"].shape[-2] - self.img_size[1])
            y1 = randint(0, sample["image"].shape[-3] - self.img_size[0])
            if self.anomalous:
                sample["mask"] = sample["mask"][x1:x1 + self.img_size[1], y1:y1 + self.img_size[0]]
            sample["image"] = sample["image"][x1:x1 + self.img_size[1], y1:y1 + self.img_size[0]]

        if self.transform:
            sample["image"] = self.transform(sample["image"])
            if self.anomalous:
                sample["mask"] = self.transform_mask(sample["mask"])
                sample["mask"] = (sample["mask"] > 0).float()

        return sample

class AnoBratsDataset(Dataset):
    """Anomalous MRI dataset."""

    def __init__(
            self, ROOT_DIR, transform=None, img_size=(256, 256), slice_selection="non_random", resized=False,
            cleaned=False
            ):
        """
        Args:
            ROOT_DIR (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            img_size: size of each 2D dataset image
            slice_selection: "random" = randomly selects a slice from the image
                             "iterateKnown" = iterates between ranges of tumour using slice data
                             "iterateUnKnown" = iterates through whole MRI volume
        """
        self.transform = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.CenterCrop((175, 240)),
                 # transforms.RandomAffine(0, translate=(0.02, 0.1)),
                 transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
                 # transforms.CenterCrop(256),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5), (0.5))
                 ]
                ) if not transform else transform
        self.img_size = img_size
        self.resized = resized
        self.slices = {
            "BraTS2021_00000": range(165, 205), "BraTS2021_00002": range(177, 213), "BraTS2021_00003": range(160, 190), "BraTS2021_00005": range(160, 212),
            "BraTS2021_00006": range(140, 200), "BraTS2021_00008": range(135, 190), "BraTS2021_00009": range(150, 205), "BraTS2021_00011": range(130, 190),
            "BraTS2021_00012": range(120, 180), "BraTS2021_00014": range(170, 194), "BraTS2021_00016": range(158, 195), "BraTS2021_00017": range(155, 195),
            "BraTS2021_00018": range(184, 213), "BraTS2021_00019": range(158, 209), "BraTS2021_00020": range(158, 210), "BraTS2021_00021": range(164, 200),
            "BraTS2021_00022": range(142, 200), "BraTS2021_00024": range(160, 200), "BraTS2021_00025": range(147, 210), "BraTS2021_00026": range(155, 200)
        }

        self.filenames = self.slices.keys()
        self.filenames = list(map(lambda name: f"{ROOT_DIR}/{name}/{name}", self.filenames))
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")
        self.ROOT_DIR = ROOT_DIR
        self.slice_selection = slice_selection
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.filenames[idx] + "_" + "t1" + ".nii.gz"
        mask_name = self.filenames[idx] + "_" + "seg" + ".nii.gz"
        img = nib.load(img_name)
        image = img.get_fdata()
        msk = nib.load(mask_name)
        mask = msk.get_fdata()            
#         image = np.rot90(image)
#         image_mean = np.mean(image)
#         image_std = np.std(image)
#         img_range = (image_mean - 1 * image_std, image_mean + 2 * image_std)
#         image = np.clip(image, img_range[0], img_range[1])
#         image = image / (img_range[1] - img_range[0])
#         np.save(
#                 os.path.join(f"{self.filenames[idx]}.npy"), image.astype(
#                         np.float32
#                         )
#                 )
        sample = {}
        if self.slice_selection == "random":
            temp_range = self.slices[self.filenames[idx].split("/")[-1]]
            slice_idx = randint(temp_range.start, temp_range.stop)
        else:
            slice_idx = 80
        image1 = image[slice_idx:slice_idx + 1, :, :].reshape(image.shape[1], image.shape[2]).astype(np.float32)
        mask = mask[slice_idx:slice_idx + 1, :, :].reshape(mask.shape[1], mask.shape[2]).astype(np.float32)
        if self.transform:
            image1 = self.transform(image1)
                # image = transforms.functional.rotate(image, -90)
        sample["slices"] = slice_idx

#         elif self.slice_selection == "iterateUnknown":

#             output = torch.empty(image.shape[0], *self.img_size)
#             for i in range(image.shape[0]):
#                 temp = image[i:i + 1, :, :].reshape(image.shape[1], image.shape[2]).astype(np.float32)
#                 if self.transform:
#                     temp = self.transform(temp)
#                     # temp = transforms.functional.rotate(temp, -90)
#                 output[i, ...] = temp

#             image = output
#             sample["slices"] = image.shape[0]

        sample["image"] = image1
        sample["mask"] = mask
        sample["filenames"] = self.filenames[idx].split("/")[-1]
#         sample = {'image': image, "filenames": self.filenames[idx], "mask":mask}
        return sample

class MRIDataset(Dataset):
    """Healthy MRI dataset."""

    def __init__(self, ROOT_DIR, transform=None, img_size=(32, 32), random_slice=False):
        """
        Args:
            ROOT_DIR (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.RandomAffine(3, translate=(0.02, 0.09)),
                 transforms.CenterCrop(235),
                 transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
                 # transforms.CenterCrop(256),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5), (0.5))
                 ]
                ) if not transform else transform

        self.filenames = os.listdir(ROOT_DIR)
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")
        self.ROOT_DIR = ROOT_DIR
        self.random_slice = random_slice

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # print(repr(idx))
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if os.path.exists(os.path.join(self.ROOT_DIR, self.filenames[idx], f"{self.filenames[idx]}.npy")):
            image = np.load(os.path.join(self.ROOT_DIR, self.filenames[idx], f"{self.filenames[idx]}.npy"))
            pass
        else:
            img_name = os.path.join(
                    self.ROOT_DIR, self.filenames[idx], f"sub-{self.filenames[idx]}_ses-NFB3_T1w.nii.gz"
                    )
            # random between 40 and 130
            # print(nib.load(img_name).slicer[:,90:91,:].dataobj.shape)
            img = nib.load(img_name)
            image = img.get_fdata()

            image_mean = np.mean(image)
            image_std = np.std(image)
            img_range = (image_mean - 1 * image_std, image_mean + 2 * image_std)
            image = np.clip(image, img_range[0], img_range[1])
            image = image / (img_range[1] - img_range[0])
            np.save(
                    os.path.join(self.ROOT_DIR, self.filenames[idx], f"{self.filenames[idx]}.npy"), image.astype(
                            np.float32
                            )
                    )
        if self.random_slice:
            # slice_idx = randint(32, 122)
            slice_idx = randint(40, 100)
        else:
            slice_idx = 80
        mask = torch.zeros(1, 256, 256)
        image = image[:, slice_idx:slice_idx + 1, :].reshape(256, 192).astype(np.float32)

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, "filenames": self.filenames[idx], "mask": mask}
        return sample
class MatTDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform=None, img_size = 256):
        
        super().__init__()
        self.transform = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.RandomAffine(3, translate=(0.02, 0.09)),
                 transforms.CenterCrop(235),
                 transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
                 # transforms.CenterCrop(256),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5), (0.5))
                 ]
                ) if not transform else transform
        self.directory = os.path.expanduser(directory)

        self.database = os.listdir(directory)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filedict = self.database[idx]
        img_name = os.path.join(
                    self.directory, self.database[idx])
        data = mat73.loadmat(img_name)["cjdata"]
        image = data["image"]
        mask = data["tumorMask"]
        label = data["label"]     
#         image_mean = np.mean(image)
#         image_std = np.std(image)
#         img_range = (image_mean - 1 * image_std, image_mean + 2 * image_std)
#         image = np.clip(image, img_range[0], img_range[1])
#         image = image / (img_range[1] - img_range[0])
#         print(image.shape)
#         print(type(image))
        image = image.astype(np.uint8)

#         # Convert npimg to PIL Image
#         image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, "mask": mask, "label": label, "filenames": self.database[idx]}
        return sample

    def __len__(self):
        return len(self.database)
class HnABRATS(torch.utils.data.Dataset):
    def __init__(self, ROOT_DIR, img_size=(256, 256), is_train = True, transform=None):     
        
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),        
             transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
             # transforms.CenterCrop(256),
             transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))
            ]
        ) if not transform else transform
        self.root = os.path.join(ROOT_DIR, "DATASETS/brats2021")
        if is_train:
            self.filenames = [i for i in open(os.path.join(self.root,'4D_1k_Healthy.txt'))]
        else:
            self.filenames = [i for i in open(os.path.join(self.root,'4D_Anomaly.txt'))]
            self.mask_filenames = [i for i in open(os.path.join(self.root,'4D_Mask.txt'))]
        self.img_size = img_size
        self.is_train = is_train
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = np.load(self.filenames[idx][:-1])
#         print(image.shape)
        if self.transform:
            for i in range(image.shape[0]):
                image1 = self.transform(image[0])
                image2 = self.transform(image[1])
                image3 = self.transform(image[2])
                image4 = self.transform(image[3])
        image = torch.stack([image1, image2, image3, image4], dim = 1)
        if not self.is_train:
            mask = np.load(self.mask_filenames[idx][:-1])
            sample = {'image': image, "filenames": self.filenames[idx].split("/")[-1][:-4], "mask":mask}
        else:
            sample = {'image': image, "filenames": self.filenames[idx].split("/")[-1][:-4]}
        return sample

    def __len__(self):
        return len(self.filenames)


class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory , test_flag=False):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('_')[3]
                    datapoint[seqtype] = os.path.join(root, f)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)

    def __getitem__(self, x):
#         out = []
        filedict = self.database[x]
#         for seqtype in self.seqtypes:
        number=filedict['t1'].split('\\')[-2]
        nib_img = nib.load(filedict["t1"])
        image = nib_img.get_fdata()
        out = image
        image_mean = np.mean(image)
        image_std = np.std(image)
        img_range = (image_mean - 1 * image_std, image_mean + 2 * image_std)
        image = np.clip(image, img_range[0], img_range[1])
        image = image / (img_range[1] - img_range[0])
#         print(image.shape)
        out_dict = {}
        if self.test_flag:
            path2 = './data/brats/test_labels/' + str(
                number) + '-label.nii.gz'


            seg=nibabel.load(path2)
            seg=seg.get_fdata()
            image = torch.zeros(1, 256, 256)
            image[:, 8:-8, 8:-8] = torch.tensor(out)
            label = seg[None, ...]
            if seg.max() > 0:
                weak_label = 1
            else:
                weak_label = 0
            out_dict["y"]=weak_label
            sample = {'image': image, "mask": seg, "y":weak_label, "label": label, "filenames": filedict['t1'].split('\\')}
        else:
            image = torch.zeros(1,256,256)
            image[:,8:-8,8:-8]=out[:-1,...]		#pad to a size of (256,256)
            label = out[-1, ...][None, ...]
            if label.max()>0:
                weak_label=1
            else:
                weak_label=0
            out_dict["y"] = weak_label
            sample = {'image': image, "y":weak_label, "label": label, "filenames": filedict}

        return sample

    def __len__(self):
        return len(self.database)



class AnomalousMRIDataset(Dataset):
    """Anomalous MRI dataset."""

    def __init__(
            self, ROOT_DIR, transform=None, img_size=(32, 32), slice_selection="iterateUnknown", resized=False,
            cleaned=False
            ):
        """
        Args:
            ROOT_DIR (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            img_size: size of each 2D dataset image
            slice_selection: "random" = randomly selects a slice from the image
                             "iterateKnown" = iterates between ranges of tumour using slice data
                             "iterateUnKnown" = iterates through whole MRI volume
        """
        self.transform = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.CenterCrop((175, 240)),
                 # transforms.RandomAffine(0, translate=(0.02, 0.1)),
                 transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR),
                 # transforms.CenterCrop(256),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5), (0.5))
                 ]
                ) if not transform else transform
        self.img_size = img_size
        self.resized = resized
        self.slices = {
            "00000": range(165, 205), "00002": range(177, 213), "00003": range(160, 190), "00005": range(160, 212),
            "00006": range(140, 200), "00008": range(135, 190), "00009": range(150, 205), "00011": range(130, 190),
            "00012": range(120, 180), "00014": range(170, 194), "00016": range(158, 195), "00017": range(155, 195),
            "00018": range(184, 213), "00019": range(158, 209), "00020": range(158, 210), "00021": range(164, 200),
            "00022": range(142, 200), "00024": range(160, 200), "00025": range(147, 210), "00026": range(155, 200)
        }

        self.filenames = self.slices.keys()
        if cleaned ==True:
            self.filenames = list(map(lambda name: f"{ROOT_DIR}/raw_cleaned/{name}.npy", self.filenames))
        elif cleaned == "brats2021":
            self.filenames = os.listdir(ROOT_DIR)
        else:
            self.filenames = list(map(lambda name: f"{ROOT_DIR}/raw/{name}.npy", self.filenames))
        # self.filenames = os.listdir(ROOT_DIR)
        if ".DS_Store" in self.filenames:
            self.filenames.remove(".DS_Store")
        self.ROOT_DIR = ROOT_DIR
        self.slice_selection = slice_selection
        self.cleaned = cleaned
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if os.path.exists(os.path.join(f"{self.filenames[idx]}")):
            if self.resized and os.path.exists(os.path.join(f"{self.filenames[idx][:-4]}-resized.npy")):
                image = np.load(os.path.join(f"{self.filenames[idx][:-4]}-resized.npy"))
            else:
                image = np.load(os.path.join(f"{self.filenames[idx]}"))
                     
        else:
            img_name = os.path.join(
                    self.ROOT_DIR, self.filenames[idx], f"{self.filenames[idx]}_t1.nii.gz"
                    )  
#             img_name = os.path.join(self.filenames[idx])
            # print(nib.load(img_name).slicer[:,90:91,:].dataobj.shape)
            img = nib.load(img_name)
            image = img.get_fdata()
            image = np.rot90(image)

            image_mean = np.mean(image)
            image_std = np.std(image)
            img_range = (image_mean - 1 * image_std, image_mean + 2 * image_std)
            image = np.clip(image, img_range[0], img_range[1])
            image = image / (img_range[1] - img_range[0])
            np.save(
                    os.path.join(f"{self.filenames[idx]}.npy"), image.astype(
                            np.float32
                            )
                    )
        sample = {}

        if self.resized:
            img_mask = np.load(f"{self.ROOT_DIR}/mask/{self.filenames[idx][-9:-4]}-resized.npy")
        elif self.cleaned =="brats2021":
            mask = nib.load(os.path.join(
                    self.ROOT_DIR, self.filenames[idx], f"{self.filenames[idx]}_seg.nii.gz"
                    )  )
            img_mask = mask.get_fdata()
        else:
            img_mask = np.load(f"{self.ROOT_DIR}/mask/{self.filenames[idx][-9:-4]}.npy")
        if self.slice_selection == "random":
            temp_range = self.slices[self.filenames[idx][-9:-4]]
            slice_idx = randint(temp_range.start, temp_range.stop)
            image = image[slice_idx:slice_idx + 1, :, :].reshape(image.shape[1], image.shape[2]).astype(np.float32)
            if self.transform:
                image = self.transform(image)
                # image = transforms.functional.rotate(image, -90)
            sample["slices"] = slice_idx
        elif self.slice_selection == "iterateKnown":
            temp_range = self.slices[self.filenames[idx][-9:-4]]
            output = torch.empty(temp_range.stop - temp_range.start, *self.img_size)
            output_mask = torch.empty(temp_range.stop - temp_range.start, *self.img_size)

            for i, val in enumerate(temp_range):
                temp = image[val, ...].reshape(image.shape[1], image.shape[2]).astype(np.float32)
                temp_mask = img_mask[val, ...].reshape(image.shape[1], image.shape[2]).astype(np.float32)
                if self.transform:
                    temp = self.transform(temp)
                    temp_mask = self.transform(temp_mask)
                output[i, ...] = temp
                output_mask[i, ...] = temp_mask

            image = output
            sample["slices"] = temp_range
            sample["mask"] = (output_mask > 0).float()

        elif self.slice_selection == "iterateKnown_restricted":

            temp_range = self.slices[self.filenames[idx][-9:-4]]
            output = torch.empty(4, *self.img_size)
            output_mask = torch.empty(4, *self.img_size)
            slices = np.linspace(temp_range.start + 5, temp_range.stop - 5, 4).astype(np.int32)
            for counter, i in enumerate(slices):
                temp = image[i, ...].reshape(image.shape[1], image.shape[2]).astype(np.float32)
                temp_mask = img_mask[i, ...].reshape(image.shape[1], image.shape[2]).astype(np.float32)
                if self.transform:
                    temp = self.transform(temp)
                    temp_mask = self.transform(temp_mask)
                output[counter, ...] = temp
                output_mask[counter, ...] = temp_mask
            image = output
            sample["slices"] = slices
            sample["mask"] = (output_mask > 0).float()

        elif self.slice_selection == "iterateUnknown":

            output = torch.empty(image.shape[0], *self.img_size)
            for i in range(image.shape[0]):
                temp = image[i:i + 1, :, :].reshape(image.shape[1], image.shape[2]).astype(np.float32)
                if self.transform:
                    temp = self.transform(temp)
                    # temp = transforms.functional.rotate(temp, -90)
                output[i, ...] = temp

            image = output
            sample["slices"] = image.shape[0]

        sample["image"] = image
        sample["filenames"] = self.filenames[idx]
        # sample = {'image': image, "filenames": self.filenames[idx], "slices":slice_idx}
        return sample


def load_CIFAR10(args, train=True):
    return torch.utils.data.DataLoader(
            datasets.CIFAR10(
                    "./DATASETS/CIFAR10", train=train, download=True, transform=transforms
                        .Compose(
                            [
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

                                ]
                            )
                    ),
            shuffle=True, batch_size=args["Batch_Size"], drop_last=True
            )
class MURA():
    def __init__(self, input_size, root = r"C:\Users\Admin\Dropbox\PC\Documents\FGVC_MSFM\MMAL-Net\datasets\MURA_DATA", is_train=True, data_len=None, subset = 'train'):
        self.input_size = input_size
        self.root = root
        self.subset = subset
        self.is_train = is_train
        train_img_path = os.path.join(self.root, 'mura_train')
        test_img_path = os.path.join(self.root, 'mura_test')
#         train_label_file = open(os.path.join(self.root, 'train_Hand.txt'))
#         test_label_file = open(os.path.join(self.root, 'test_Hand.txt'))
#         train_label_file = open(os.path.join(self.root, 'train.txt'))
#         test_label_file = open(os.path.join(self.root, 'test.txt'))
        train_label_file = open(os.path.join(self.root, 'train_' + self.subset +'.txt'))
        test_label_file = open(os.path.join(self.root, 'test_' + self.subset + '.txt'))
        train_img_label = []
        test_img_label = []
        for line in train_label_file:
            train_img_label.append([os.path.join(train_img_path, line[:-2]), int(line[-2])])
        for line in test_label_file:
            test_img_label.append([os.path.join(test_img_path, line[:-2]), int(line[-2])])
        self.train_img_label = train_img_label[:data_len]
        self.test_img_label = test_img_label[:data_len]


    def __getitem__(self, index):
        if self.is_train:
            img, target = imageio.imread(self.train_img_label[index][0]), self.train_img_label[index][1]
#             print(target)
            filenames = self.train_img_label[index][0]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')

            img = transforms.Resize((self.input_size, self.input_size))(img)
            img = transforms.RandomResizedCrop(size=self.input_size,scale=(0.4, 0.75),ratio=(0.5,1.5))(img)
            img = transforms.RandomCrop(self.input_size)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)

            img = transforms.ToTensor()(img)
            img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
#             img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = imageio.imread(self.test_img_label[index][0]), self.test_img_label[index][1]
            filenames = self.train_img_label[index][0]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
            # img = transforms.CenterCrop(self.input_size)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
#             img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        
        sample = {'image': img, "filenames": filenames, "label": target}
        return sample


    def __len__(self):
        if self.is_train:
            return len(self.train_img_label)
        else:
            return len(self.test_img_label)

class Pneumonia():
    def __init__(self, input_size = 256, root = r"C:\Users\Admin\Dropbox\PC\Documents\Anomaly Detection\AnoDDPM\DATASETS\CheNemonia\chest_xray\train\NORMAL", data_len=None, is_train = True):
        self.input_size = input_size
        self.root = root
        self.database = os.listdir(self.root)
        self.is_train = is_train

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.database[index])
#         if self.is_train:
        img = imageio.imread(image_path)
#             print(target)
        filenames = self.database[index]
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        img = transforms.Resize((self.input_size, self.input_size))(img)
        if self.is_train:
            img = transforms.RandomResizedCrop(size=self.input_size,scale=(0.4, 0.75), ratio=(0.5,1.5))(img)
            img = transforms.RandomCrop(self.input_size)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        mask = torch.zeros(3, self.input_size, self.input_size)
#             img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        sample = {'image': img, "filenames": filenames, "mask":mask}
        return sample


    def __len__(self):
        return len(self.database)
class TUMOR():
    def __init__(self, input_size, root = r"C:\Users\Admin\Dropbox\PC\Documents\FGVC_MSFM\MMAL-Net\datasets", is_train=True, data_len=None):
        self.input_size = input_size
        self.root = root
#         self.transform = transform
#         self.to_pil = transforms.ToPILImage()
        self.is_train = is_train
        train_img_path = os.path.join(self.root)
        test_img_path = os.path.join(self.root)
        train_label_file = open(os.path.join(self.root, 'Tumor\\train_negative.txt'))
        test_label_file = open(os.path.join(self.root, 'Tumor\\test.txt'))
        train_img_label = []
        test_img_label = []
        for line in train_label_file:
            train_img_label.append([os.path.join(train_img_path, line[:-2]), int(line[-2])])
        for line in test_label_file:
            test_img_label.append([os.path.join(test_img_path, line[:-2]), int(line[-2])])
        self.train_img_label = train_img_label[:data_len]
        self.test_img_label = test_img_label[:data_len]


    def __getitem__(self, index):
        if self.is_train:
            img, target = imageio.imread(self.train_img_label[index][0]), self.train_img_label[index][1]
            filenames = self.train_img_label[index][0]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img)

            img = transforms.Resize(self.input_size, Image.Resampling.BILINEAR)(img)
            # img = transforms.RandomResizedCrop(size=self.input_size,scale=(0.4, 0.75),ratio=(0.5,1.5))(img)
            # img = transforms.RandomCrop(self.input_size)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)

            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = imageio.imread(self.test_img_label[index][0]), self.test_img_label[index][1]
            filenames = self.test_img_label[index][0]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize(self.input_size, Image.BILINEAR)(img)
            # img = transforms.CenterCrop(self.input_size)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        sample = {'image': img, "filenames": filenames, "label": target}
        return sample

    def __len__(self):
        if self.is_train:
            return len(self.train_img_label)
        else:
            return len(self.test_img_label)

class AnoClsDataset(Dataset):
    def __init__(self, ROOT_DIR, img_size, transform=None, data_len=None):
        self.input_size = img_size
        self.root = ROOT_DIR
        l1  = [[os.path.join(self.root, 'no', x), 0, x] for x in os.listdir(os.path.join(self.root, "no"))]
        l2  = [[os.path.join(self.root, 'yes', x), 1, x] for x in os.listdir(os.path.join(self.root, "yes"))]
        self.img_label = l1+l2

    def __getitem__(self, index):
            img, target, filename = cv2.imread(self.img_label[index][0], cv2.IMREAD_GRAYSCALE), self.img_label[index][1], self.img_label[index][2]
            img = Image.fromarray(img)
#             img = np.asarray(img)
#             if self.transform:
#                 img = self.transform(img)
            img = transforms.Resize(self.input_size, Image.BILINEAR)(img)
            # img = transforms.CenterCrop(self.input_size)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize(0.5, 0.5)(img)
            sample = {'image': img, "label": target, "filenames": filename}
            return sample

    def __len__(self):
        return len(self.test_img_label)
if __name__ == "__main__":
    # load_datasets_for_test()
    # get_segmented_labels(True)
    # main(False, False, 0)
    # make_pngs_anogan()
    import matplotlib.pyplot as plt
    import helpers

    d_set = MVTec(
            './DATASETS/leather', True, img_size=(256, 256), rgb=False
            )
    # d_set = AnomalousMRIDataset(
    #         ROOT_DIR='./DATASETS/CancerousDataset/EdinburghDataset/Anomalous-T1', img_size=(256, 256),
    #         slice_selection="iterateKnown_restricted", resized=False
    #         )
    loader = init_dataset_loader(d_set, {"Batch_Size": 16})

    for i in range(4):
        new = next(loader)
        plt.imshow(helpers.gridify_output(new["image"], 4), cmap="gray")
        plt.show()
        plt.imshow(helpers.gridify_output(new["mask"], 4), cmap="gray")
        plt.show()
        plt.pause(1)
