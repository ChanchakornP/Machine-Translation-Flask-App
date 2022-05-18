import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer_th2en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-th-en")
model_th2en = AutoModelForSeq2SeqLM.from_pretrained('model_final_th2en')
tokenizer_en2th = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-mul')
model_en2th = AutoModelForSeq2SeqLM.from_pretrained('model_final_en2th')

def pred_th2en(words):
    try:
        input_ids = torch.tensor(tokenizer_th2en.encode(words, add_special_tokens=True)).unsqueeze(0)
        output_ids = model_th2en.generate(input_ids,decoder_start_token_id=model_th2en.config.pad_token_id)
        outputs = tokenizer_th2en.decode(output_ids[0], skip_special_tokens=True)
    except:
        return 0, 'error'
    return outputs

def pred_en2th(words):
    try:
        input_ids = torch.tensor(tokenizer_en2th.encode(words, add_special_tokens=True)).unsqueeze(0)
        output_ids = model_en2th.generate(input_ids,decoder_start_token_id=model_en2th.config.pad_token_id)
        outputs = tokenizer_en2th.decode(output_ids[0], skip_special_tokens=True)
    except:
        return 0, 'error'
    return outputs
