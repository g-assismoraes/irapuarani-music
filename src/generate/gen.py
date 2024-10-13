from huggingface_hub import login
import argparse
import transformers
import pandas as pd
import torch
import json
import gc
import os

def main(model_id, hf_token, input_path, output_path):
    login(token=hf_token)
    
    
    data = []
    for filename in os.listdir(input_path):
        if filename.endswith('.txt'):  
            file_path = os.path.join(input_path, filename)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            data.append({'file_name': filename, 'content': content})
        
    df = pd.DataFrame(data)

    device = f'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    pipeline = transformers.pipeline(
        task="text-generation",
        trust_remote_code=True,
        model=model_id,
        tokenizer=tokenizer,
        # The quantization line
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device,
    )


    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    
    def get_prompt(music):
        return  f'''You will classify music represented in symbolic form as either positive or negative.

### Symbolic Representation
d_[duration]_[dots]: Defines the duration of the upcoming notes. The [duration] specifies the type of note (e.g., breve, whole, half, quarter, eighth, 16th, or 32nd). The [dots] indicates the number of dots extending the note’s duration, and can be any integer from 0 to 3.
v_[velocity]: Indicates the velocity (or loudness) of the following notes. Velocity is discretized into bins of size 4, allowing values such as 4, 8, 12, up to 128.
t_[tempo]: Changes the tempo of the piece, measured in beats per minute (bpm). Tempo is discretized into bins of size 4, ranging from 24 to 160 bpm. This controls the speed at which the piece is played.
w_[wait]: Specifies the number of time steps (units of waiting) that pass before the next musical event occurs. The value associated with w, such as in w_2 or w_3, represents the number of time steps with no musical events.
\n: Marks the end of the piece.

### Music
{music}


Your answer must strictly follow this format:
- answer: A string, either "positive" or "negative"
- justify: A brief explanation justifying your classification
'''

    results = []
    for idx, row in df.iterrows():
        file_name = row['file_name']
        music_content = row['content'][:1800] 
        
        
        messages = [
            {"role": "user", "content": get_prompt(music_content)}
        ]
        
        outputs = pipeline(
            messages,
            max_new_tokens=500,
            do_sample=True,
            temperature=1,
            top_p=0.95,
        )

        generated_texts = outputs[0]["generated_text"][1]['content']
        
        result = {
            "file_name": file_name,
            "music_passed_to_model": music_content,
            "model_output": generated_texts
        }
        
        results.append(result)
        
        del outputs
        del messages
        torch.cuda.empty_cache()
        gc.collect()
        
        os.makedirs(output_path, exist_ok=True)
        output_name = model_id
        if '/' in output_name:
            output_name = output_name.split('/')[-1]
            
        json_output_path = os.path.join(output_path, f"{output_name}_temp1.json")
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run text generation with Hugging Face pipeline.")
    parser.add_argument('--model_id', type=str, required=True, help="Hugging Face Model ID.")
    parser.add_argument('--hf_token', type=str, required=True, help="Hugging Face API token.")
    parser.add_argument('--input_path', type=str, required=True, help="Path to get the txt inputs.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the generated outputs.")

    args = parser.parse_args()


    main(args.model_id, args.hf_token, args.input_path, args.output_path)