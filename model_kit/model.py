from transformers import VisionEncoderDecoderModel
from model_kit.utils import processor

processor = processor()


def tr_ocr_model(model_config, device):
    model = VisionEncoderDecoderModel.from_pretrained(model_config)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} trainable parameters.")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.eos_token_id

    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    return model
