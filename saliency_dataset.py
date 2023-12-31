import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class SaliencyDataset(Dataset):
    def __init__(self, csv_file, source_image_dir, target_image_dir, target_fixation_dir, output_type,image_size, transform=None, split="train"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            source_image_dir (string): Directory with all the source images.
            target_image_dir (string): Directory with all the target images.
            output_type (int): The type of data to include in the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file, usecols=['image', 'text', 'type'])
        self.source_image_dir = source_image_dir
        self.target_image_dir = target_image_dir
        self.target_fixation_dir = target_fixation_dir
        self.image_size = image_size # should be tuple 
        self.output_type = output_type
        self.transform = transform or transforms.Compose([
                                        transforms.Resize(self.image_size),
                                        transforms.ToTensor()
                                    ])
        self.split = split

        # Filter data_frame for rows with the specified output_type
        self.pair_image_names = None
        
        # Filter target image names based on output_type
        if output_type != 10:
            self.filtered_data_frame = self.data_frame[self.data_frame['type'] == self.output_type]
            self.target_image_names = [name for name in os.listdir(self.target_image_dir)
                                    if self._has_matching_type(name)]
            if split=="train":
                # take 80% of the data
                self.target_image_names = self.target_image_names[:int(len(self.target_image_names)*0.8)]
            if split == "val":
                # take 20% last of the data
                self.target_image_names = self.target_image_names[int(len(self.target_image_names)*0.8):]
        else:
            self.filtered_data_frame = self.data_frame
            self.pair_image_names = [(name, name, name,"", 10) for name in os.listdir(self.source_image_dir)]
            self.out = []

            for idx in range(len(self.pair_image_names)):
                
                source, target, fixation, text, out_type = self.pair_image_names[idx]


                # out_put type determine
                pre, _, suffix = source.partition('_')
                pre = pre.lstrip('0')  # Remove leading zeros
                if not suffix:
                    output_type == 4  # type 0 corresponds to no suffix
                    pre = pre.split('.')[0]
                # Extract the numeric suffix and check against output_type
                else:
                    suffix = suffix.split('.')[0]
                    output_type == int(suffix)

                # text retrival
                
                pre_int = int(pre)
                # print(pre_int, suffix )
                row = self.filtered_data_frame[self.filtered_data_frame['image'] == pre_int]
                text = row.iloc[0]['text'] if not row.empty else ""
                # print(text)
                if text=="":
                    continue

                # Create datapoint
                # datapoint = (source_image, target_image, target_fixation_img, text, self.output_type)


                # image read
                source_image = Image.open(self.source_image_dir+source).convert('RGB')
                if target:
                    target_image = Image.open(self.target_image_dir+ target).convert('RGB')
                if fixation:
                    target_fixation_img = Image.open(self.target_fixation_dir+fixation)

                # Apply transforms if any
                if self.transform:
                    source_image = self.transform(source_image)
                    target_image = self.transform(target_image)[:1,:,:]
                    target_fixation_img = self.transform(target_fixation_img)[:1,:,:]

                self.out.append((source_image, target_image, target_fixation_img, text, output_type))

            if split == "train":
                # take 80% of the data
                self.out = self.out[:int(len(self.out)*0.8)]
            elif split == "val":
                # take 20% of the data
                self.out = self.out[int(len(self.out)*0.8):]

                    

    def _has_matching_type(self, image_name):
        # Split name and suffix
        pre, _, suffix = image_name.partition('_')
        if not suffix:

            return self.output_type == 4  # type 0 corresponds to no suffix
        
        # Extract the numeric suffix and check against output_type
        suffix = suffix.split('.')[0]
        return int(suffix) == self.output_type

    def __len__(self):
        if self.output_type !=10:
            return len(self.target_image_names) 
        else:
            return len(self.out)

    def __getitem__(self, idx):
        # Get the name of the target image
        if self.output_type !=10:
            target_image_name = self.target_image_names[idx]
            
            # Separate name and suffix
            pre, _, _ = target_image_name.partition('_')
            pre = pre.lstrip('0')  # Remove leading zeros

            if self.output_type==4:
                pre = pre.split('.')[0]

            # Get corresponding source image path and target image path
            source_image_path = os.path.join(self.source_image_dir, target_image_name)
            target_image_path = os.path.join(self.target_image_dir, target_image_name)
            target_fixation_path = os.path.join(self.target_fixation_dir, target_image_name)

            # Load images
            source_image = Image.open(source_image_path).convert('RGB')
            target_image = Image.open(target_image_path).convert('RGB')
            target_fixation_img = Image.open(target_fixation_path)

            # Apply transforms if any
            if self.transform:
                source_image = self.transform(source_image)
                target_image = self.transform(target_image)[:1,:,:]
                target_fixation_img = self.transform(target_fixation_img)[:1,:,:]

            # Fetch the appropriate row
            pre_int = int(pre)
            row = self.filtered_data_frame[self.filtered_data_frame['image'] == pre_int]
            text = row.iloc[0]['text'] if not row.empty else ""

            # Create datapoint
            datapoint = (source_image, target_image, target_fixation_img, text, self.output_type)

            return datapoint
        else:
            datapoint = self.out[idx]

            return datapoint


