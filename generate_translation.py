from transformers import pipeline,AutoModel,AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import jsonlines

# The Wanderer ranges from 0 to 3361
# Load data
raw = load_dataset("/home/maxime/Documents/LUMIA/Project/transformers-translation/opus_books","en-fr",split="train")
the_wanderer_data = raw[0:3362]["translation"]

# Load model
model_checkpoint = "/home/maxime/Documents/LUMIA/Project/transformers-translation/opus-mt-fr-en"
translator = pipeline("translation", model=model_checkpoint)


# Generate results
output_file = "/home/maxime/Documents/LUMIA/Project/translation/generated_result.jsonl"
with jsonlines.open(output_file, mode="w") as file:
    for data_line in tqdm(the_wanderer_data, desc="fr-en translation"):
        result = {}
        result['fr'] = data_line["fr"]
        result['label'] = data_line["en"]
        result['pred'] = translator(data_line["fr"])[0]["translation_text"]
        
        file.write(result)