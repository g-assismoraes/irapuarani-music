import time
import torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, 
                          Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorForSeq2Seq)
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig, TaskType
from transformers import DataCollatorWithPadding
from accelerate import Accelerator
from transformers import AdamW
from huggingface_hub import login
import numpy as np
import os
import argparse
import eco2ai
import pandas as pd
import json
from sklearn.model_selection import KFold

# Disable NCCL features for compatibility
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

def main(src_model, model_name, hub_id, output_file, input_json, hug_token=''):
    
    # CO2 Tracking
    tracker = eco2ai.Tracker(
        project_name=f'''{model_name} Training''', 
        experiment_description="Training multiple LLMs with 10-fold cross-validation",
        file_name=output_file.replace("txt", "csv")
    )
    
    # Login to Hugging Face Hub
    login(token=hug_token)
    
    # Load dataset from CSV
    df = pd.read_csv(input_json)

    # Mapping valence to labels
    df['label'] = df['valence'].apply(lambda x: "positive" if x == 1 else "negative")
    
    # Create the Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    
    
    def get_prompt(music):
        return  f'''You will classify music represented in symbolic form as either positive or negative.

### Symbolic Representation
d_[duration]_[dots]: Defines the duration of the upcoming notes. The [duration] specifies the type of note (e.g., breve, whole, half, quarter, eighth, 16th, or 32nd). The [dots] indicates the number of dots extending the note’s duration, and can be any integer from 0 to 3.
v_[velocity]: Indicates the velocity (or loudness) of the following notes. Velocity is discretized into bins of size 4, allowing values such as 4, 8, 12, up to 128.
t_[tempo]: Changes the tempo of the piece, measured in beats per minute (bpm). Tempo is discretized into bins of size 4, ranging from 24 to 160 bpm. This controls the speed at which the piece is played.
n_[pitch]: Specifies the pitch of a note using its MIDI pitch number. The [pitch] value is an integer ranging from 0 to 127, representing the full range of MIDI pitches. For example, n_60 corresponds to Middle C.
w_[wait]: Specifies the number of time steps (units of waiting) that pass before the next musical event occurs. The value associated with w, such as in w_2 or w_3, represents the number of time steps with no musical events.
\n: Marks the end of the piece.

### Music
{music}

Your answer must strictly follow this format:
- answer: A string, either "positive" or "negative"'''
    
    # Preprocessing function for the classification task
    def preprocess_function(examples):
        inputs = [get_prompt(ex) for ex in examples['music']]
        model_inputs = tokenizer(inputs, max_length=1000, truncation=True, padding='max_length')
        
        # Tokenize labels and replace padding token id with -100
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['label'], max_length=1000, truncation=True, padding='max_length')
            labels["input_ids"] = [
                [(label if label != tokenizer.pad_token_id else -100) for label in label_seq]
                for label_seq in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    # Load tokenizer and model based on model type
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if 'cabrita' in model_name or 'llama' in model_name:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = load_model_based_on_type(model_name, tokenizer)
    
    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 10-fold cross-validation setup
    kf = KFold(n_splits=10)
    results = []
    all_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"Starting fold {fold+1}...")

        # Split data into training and validation sets
        train_data = df.iloc[train_idx].reset_index(drop=True)
        val_data = df.iloc[val_idx].reset_index(drop=True)

        # Convert to Hugging Face Dataset
        train_dataset = Dataset.from_pandas(train_data)
        val_dataset = Dataset.from_pandas(val_data)

        # Save the raw training and validation datasets for this fold
        fold_dir = f"{hub_id}_fold_{fold+1}"
        os.makedirs(fold_dir, exist_ok=True)
        train_dataset.to_json(os.path.join(fold_dir, f"train_dataset_fold_{fold+1}.json"))
        val_dataset.to_json(os.path.join(fold_dir, f"val_dataset_fold_{fold+1}.json"))

        # Preprocess datasets
        tokenized_train = train_dataset.map(preprocess_function, batched=True)
        tokenized_val = val_dataset.map(preprocess_function, batched=True)

        tokenized_train = tokenized_train.remove_columns(['series', 'console', 'game', 'piece', 'music', 
                                                          'valence', 'valence_gemma', 'valence_mistral', 
                                                          'valence_llama', 'valence_phi', 'valence_qwen', 
                                                          'valence_gpt', 'label'])
        tokenized_val = tokenized_val.remove_columns(['series', 'console', 'game', 'piece', 'music', 
                                                      'valence', 'valence_gemma', 'valence_mistral', 
                                                      'valence_llama', 'valence_phi', 'valence_qwen', 
                                                      'valence_gpt', 'label'])
        
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')

        
        optimizer = AdamW(model.parameters(), lr=2e-5)
        accelerator = Accelerator()
        
        model, optimizer, tokenized_train, tokenized_val = accelerator.prepare(
            model, optimizer, tokenized_train, tokenized_val
        )

        # Training arguments
        training_args = TrainingArguments(
            evaluation_strategy="epoch",  
            learning_rate=2e-5,
            logging_steps=50,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=3,  
            weight_decay=0.01,
            save_strategy="epoch",
            save_total_limit=1,
            fp16=False,
            push_to_hub=True, 
            output_dir=fold_dir,
            report_to="none",
            remove_unused_columns=False,
            eval_accumulation_steps=100,
            load_best_model_at_end=True,  
            metric_for_best_model="eval_loss",  
            greater_is_better=False,  
        )

        # Initialize the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            optimizers=(optimizer, None)
        )

        # Start training and measure time
        tracker.start()
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        total_training_time = end_time - start_time
        tracker.stop()

        # Evaluate the model
        metrics = trainer.evaluate()
        metrics["training_time"] = total_training_time
        metrics["fold"] = fold + 1
        results.append(metrics)
        all_metrics.append(metrics)

        # Save the model and metrics for the current fold
        trainer.save_model(fold_dir)
        with open(os.path.join(fold_dir, f"metrics_fold_{fold+1}.json"), "w") as metrics_file:
            json.dump(metrics, metrics_file)

        print(f"Fold {fold+1} results: {metrics}")

    # Compute average results across all folds
    avg_results = {key: np.mean([result[key] for result in results]) for key in results[0]}
    print(f"Average cross-validation results: {avg_results}")

    # Save the average metrics
    with open(os.path.join(output_file, "average_metrics.json"), "w") as avg_metrics_file:
        json.dump(avg_results, avg_metrics_file)

def load_model_based_on_type(model_name, tokenizer):
    return AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True, torch_dtype="bfloat16")
    # else:
    #     raise ValueError(f"Model type for {model_name} is not supported.")

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Fine-tune various LLMs with 10-fold cross-validation.")
    parser.add_argument('--src_model', type=str, required=False, help='Source model name')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--hub_id', type=str, required=True, help='Hugging Face output hub repository ID')
    parser.add_argument('--output', type=str, required=True, help='Output file to save training time')
    parser.add_argument('--input_json', type=str, required=True, help='Path to the input JSON file')
    parser.add_argument('--hug_token', type=str, required=False, help='Hugging Face token for authentication')
    
    args = parser.parse_args()
    
    # Execute main function with provided arguments
    main(args.src_model, args.model, args.hub_id, args.output, args.input_json, args.hug_token)

