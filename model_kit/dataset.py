from torch.utils.data import Dataset
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, root_dir, df, processor, transforms, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['image'][idx]

        text = self.df['label'][idx]

        image = Image.open(self.root_dir + file_name).convert('RGB')
        image = self.transforms(image)
        pixel_values = self.processor(image, return_tensors='pt').pixel_values

        labels = self.processor.tokenize(text,
                                         padding="max_length",
                                         max_length=self.max_target_length
                                         ).input_ids
        
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values, "labels": labels}
        return encoding
