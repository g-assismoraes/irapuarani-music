from huggingface_hub import login
import argparse
import transformers
import torch
import json
import gc
import os

def main(hf_token, output_path):
    login(token=hf_token)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for num in range(1, 11):
        # Model and input file for the current fold
        model_id = f"g-assismoraes/llama-music_fold_{num}"
        input_file = f"val_dataset_fold_{num}.jsonl"

        # Load the model and tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct', trust_remote_code=True)
        pipeline = transformers.pipeline(
            task="text-generation",
            trust_remote_code=True,
            model=model_id,
            tokenizer=tokenizer,
            # The quantization line
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device,
        )

        # Load the validation dataset
        data = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))

        # Function to create the prompt for each "music"
        def get_prompt(music):
            return f'''You will classify music represented in symbolic form as either positive or negative.

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

        # Run inference and collect results
        results = []
        for item in data:
            music_content = item['music']
            
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

            # Extract the generated answer
            generated_texts = outputs[0]["generated_text"][1]['content']

            # Add the result to the item with the new key "llama3B_valence"
            item["llama3B_valence"] = generated_texts

            # Append the updated item to results
            results.append(item)

            # Clean up memory
            torch.cuda.empty_cache()
            gc.collect()

        # Save the results to the output path
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"val_predictions_fold_{num}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"Finished processing fold {num}. Results saved to {output_file}.")

        # Clean up
        del pipeline
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on music classification models for 10 folds.")
    parser.add_argument('--hf_token', type=str, required=True, help="Hugging Face API token.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the generated outputs.")

    args = parser.parse_args()

    main(args.hf_token, args.output_path)
