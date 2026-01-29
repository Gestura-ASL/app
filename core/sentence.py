import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

class SentenceTranslator():
    def __init__(self) -> None:
        self.word_buffer = []
        self.current_sentence = ""
        self.model_name =  "mrm8488/t5-base-finetuned-common_gen"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)

        self.device = torch.device("mps")
        self.model.to(self.device)

    def get_display_string(self):
        return self.current_sentence

    def generate_sentence(self):
        input_text = " ".join(self.word_buffer)
    
        features = self.tokenizer([input_text], return_tensors='pt').to(device)

        # Generate with slightly better parameters for sentence variety
        output = self.model.generate(
            input_ids=features['input_ids'], 
            attention_mask=features['attention_mask'],
            max_length=256,
            num_beams=5,
            repetition_penalty=2.5,
            early_stopping=True
        )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def add_word(self,word):
        self.word_buffer.append(word)
    