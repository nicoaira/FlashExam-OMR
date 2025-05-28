from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def recognize_name_id(name_img_path, id_img_path, device='cpu'):
    # Load model and processor only once (cache as global)
    global _trocr_model, _trocr_processor, _trocr_device
    if '_trocr_model' not in globals() or _trocr_device != device:
        _trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        _trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten').to(device)
        _trocr_device = device
    def ocr(img_path):
        image = Image.open(img_path).convert('RGB')
        pixel_values = _trocr_processor(images=image, return_tensors="pt").pixel_values.to(device)
        generated_ids = _trocr_model.generate(pixel_values)
        text = _trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()
    name_text = ocr(name_img_path)
    id_text = ocr(id_img_path)
    return name_text, id_text
