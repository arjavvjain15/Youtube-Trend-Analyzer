# train_summarizer.py (Corrected Version 4)

import nltk
import numpy as np
from datasets import load_from_disk
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

def main():
    # =================================================================
    # 1. SETUP & CONFIGURATION
    # =================================================================
    print("Downloading NLTK data packages...")
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    # --- Model and Path Configuration ---
    model_checkpoint = "sshleifer/distilbart-cnn-12-6"
    local_dataset_path = "./local_cnn_dailymail"
    local_model_dir = "distilbart-cnn-dailymail-finetuned"

    # =================================================================
    # 2. LOAD DATASET AND MODEL
    # =================================================================
    print(f"Loading dataset from local disk at: {local_dataset_path}")
    raw_datasets = load_from_disk(local_dataset_path)

    # --- QUICK TRIAL RUN (Optional) ---
    print("--- Using TRIAL MODE with a small data subset ---")
    raw_datasets["train"] = raw_datasets["train"].select(range(400))
    raw_datasets["validation"] = raw_datasets["validation"].select(range(40))
    raw_datasets["test"] = raw_datasets["test"].select(range(40))
    # ------------------------------------

    print(f"Loading tokenizer and model from '{model_checkpoint}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    # =================================================================
    # 3. PREPROCESSING
    # =================================================================
    max_input_length = 1024
    max_target_length = 128

    def preprocess_function(examples):
        inputs = ["summarize: " + doc for doc in examples["article"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        labels = tokenizer(text_target=examples["highlights"], max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing the dataset (this might take a while)...")
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=["article", "highlights", "id"])

    # =================================================================
    # 4. EVALUATION METRIC (ROUGE)
    # =================================================================
    metric = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        
        # CHANGE: Simplified the result processing to handle the new output format
        result = {key: value * 100 for key, value in result.items()}
        
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        
        return {k: round(v, 4) for k, v in result.items()}

    # =================================================================
    # 5. TRAINING
    # =================================================================
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=local_model_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print(f"\nTraining finished. Model saved to '{local_model_dir}'.")
    trainer.save_model(local_model_dir)

if __name__ == "__main__":
    main()