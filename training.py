from model_kit.utils import processor, check_cuda_available
from model_kit.dataset import CustomDataset
from torchvision.transforms import transforms
from model_kit.config import DatasetConfig, ModelConfig, TrainingConfig
from model_kit.model import tr_ocr_model
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, default_data_collator
from model_kit.metrics import compute_cer
from torch.optim import AdamW
import pandas as pd
import os

if __name__ == '__main__':
    processor = processor()
    model = tr_ocr_model(model_config=ModelConfig.model_name, device=check_cuda_available())

    train_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=.5, hue=.3),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    ])

    train_df = pd.read_fwf(
        os.path.join(DatasetConfig.data_root, "scut_train.txt"), header=None
    )

    train_df.rename(columns={0: "image", 1: "label"}, inplace=True)

    test_df = pd.read_fwf(
        os.path.join(DatasetConfig.data_root, "scut_test.txt"), header=None
    )
    test_df.rename(columns={0: "image", 1: "label"}, inplace=True)

    train_dataset = CustomDataset(
        root_dir=os.path.join(DatasetConfig.data_root, "scut_train/"),
        df=train_df,
        processor=processor,
        transforms=train_transforms,
        max_target_length=512
    )

    valid_dataset = CustomDataset(
        root_dir=os.path.join(DatasetConfig.data_root, "scut_test/"),
        df=test_df,
        processor=processor,
        transforms=train_transforms,
        max_target_length=512
    )

    optimizer_metric = AdamW(model.parameters(), lr=TrainingConfig.learning_rate, weight_decay=0.0005)

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=TrainingConfig.batch_size,
        per_device_eval_batch_size=TrainingConfig.batch_size,
        fp16=True,
        output_dir='checkpoints/',
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=5,
        num_train_epochs=TrainingConfig.epochs,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor,
        args=training_args,
        compute_metrics=compute_cer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=default_data_collator
    )

    trainer.train()
