import torch

from datasets import load_dataset

from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

# =========================
# TOKENIZER
# =========================

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token

# =========================
# SMALL GPT CONFIG
# =========================

config = GPT2Config(

    vocab_size=tokenizer.vocab_size,

    n_positions=128,
    n_ctx=128,

    n_embd=128,

    n_layer=2,
    n_head=4
)

model = GPT2LMHeadModel(config)

# =========================
# LOAD DATASET
# =========================

dataset = load_dataset(

    "text",

    data_files={"train": "expanded_sentences.txt"}
)

# =========================
# TOKENIZE
# =========================

def tokenize_function(examples):

    return tokenizer(

        examples["text"],

        truncation=True,

        padding="max_length",

        max_length=128
    )

tokenized_dataset = dataset.map(

    tokenize_function,

    batched=True,

    remove_columns=["text"]
)

# =========================
# DATA COLLATOR
# =========================

data_collator = DataCollatorForLanguageModeling(

    tokenizer=tokenizer,

    mlm=False
)

# =========================
# TRAINING
# =========================

training_args = TrainingArguments(

    output_dir="./mini_gpt",

    num_train_epochs=10,

    per_device_train_batch_size=8,

    save_steps=500,

    save_total_limit=2,

    logging_steps=50,

    learning_rate=5e-4,

    prediction_loss_only=True
)

# =========================
# TRAINER
# =========================

trainer = Trainer(

    model=model,

    args=training_args,

    data_collator=data_collator,

    train_dataset=tokenized_dataset["train"]
)

# =========================
# TRAIN
# =========================

trainer.train()

# =========================
# SAVE
# =========================

model.save_pretrained("./mini_gpt")

tokenizer.save_pretrained("./mini_gpt")

print("\nTRAINING COMPLETE")
