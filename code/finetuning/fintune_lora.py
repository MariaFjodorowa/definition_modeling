# based on https://github.com/FrancescoPeriti/MultilingualDefGen/blob/main/src/finetuning.py
# https://www.kaggle.com/code/aisuko/fine-tuning-t5-small-with-lora

import torch
import random
import argparse
import evaluate
import nltk
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from accelerate import FullyShardedDataParallelPlugin, Accelerator, PartialState
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig, Seq2SeqTrainer, \
    Seq2SeqTrainingArguments, EarlyStoppingCallback, DataCollatorForSeq2Seq


def decode(text):
    try:
        return text.encode().decode('unicode_escape')
    except:
        return text


def train(args):
    if args.verbose: print('-- Set seed --')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.verbose: print(f'-- Set accelerator --')
    fsdp_plugin = FullyShardedDataParallelPlugin(  # see: https://huggingface.co/docs/accelerate/v0.11.0/en/fsdp
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False))
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

    output_dir = Path(f'{args.output_dir}/{args.finetuned_model_name}{args.tag}')
    logging_dir = str(output_dir.parent) + f'/log_{output_dir.name}'
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        auto_find_batch_size=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_dir=logging_dir,
        logging_strategy="steps",
        logging_steps=1000,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        push_to_hub=False,
        report_to="wandb",
        overwrite_output_dir=True,
        predict_with_generate=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        optim="adamw_bnb_8bit",
        metric_for_best_model="eval_rouge1",
        save_only_model=True,
        generation_max_length=24,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    if args.verbose: print('-- Load tokenizer --')
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)

    if args.verbose: print('-- Load train dataset --')
    data_files = {}
    data_files["train"] = args.train_filename
    data_files["validation"] = args.dev_filename
    raw_datasets = load_dataset(
        "csv",
        sep="\t",
        data_files=data_files,
    )

    def preprocess_func(data):
        # tokenize each row of inputs and outputs
        model_inputs = tokenizer(data["example"], truncation=True, max_length=args.max_seq_length, padding='max_length')
        labels = tokenizer(data["definition"], truncation=True, max_length=24, padding='max_length')

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # We tokenize the entire dataset
    train_dataset = raw_datasets["train"].map(preprocess_func, batched=True)
    eval_dataset = raw_datasets["validation"].map(preprocess_func, batched=True)

    if args.verbose: print(f'-- Set tuning parameters [model, device, cache] --')
    settings = dict(pretrained_model_name_or_path=args.base_model_name,
                    device_map='auto',
                    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # enable 4-bit quantization
        bnb_4bit_use_double_quant=True,  # enables double quantization (speed-up finetuning)
        bnb_4bit_quant_type="nf4",  # specifies the type of 4-bit quantization
        bnb_4bit_compute_dtype=torch.float16,  # specifies the data type for computation
    )
    settings['quantization_config'] = bnb_config

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q", "v"],
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )

    if args.verbose: print(f'-- Load base model --')
    base_model = AutoModelForSeq2SeqLM.from_pretrained(**settings)
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.use_cache = False  # avoid using cache params
    base_model.gradient_checkpointing_enable()  # this will reduce GPU memory but slow down the process
    base_model = prepare_model_for_kbit_training(
        base_model)  # see: https://huggingface.co/docs/transformers/v4.18.0/en/performance#gradient-checkpointing
    base_model.config.pretraining_tp = 1  # info: https://github.com/huggingface/transformers/pull/24906
    model = get_peft_model(base_model, peft_config)
    model = accelerator.prepare_model(model, device_placement=True)
    model.print_trainable_parameters()

    if args.verbose: print(f'-- Set Trainer --')

    # ignore tokenizer pad token in the loss
    label_pad_token_id = -100

    # padding the sentence of the entire datasets
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    metric = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100 in the predictions as we can't decode them.
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, tokenizer=lambda x: x.split())
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    if args.verbose: print(f'-- Training is started! --')
    trainer.train()

    if args.verbose: print(f'-- Store final model --')
    trainer.model.save_pretrained(str(output_dir) + f'/final-epoch{args.num_train_epochs}')
    trainer.tokenizer.save_pretrained(str(output_dir) + f'/final-epoch{args.num_train_epochs}')
    pd.DataFrame(trainer.state.log_history).to_csv(str(output_dir) + f'/log.tsv', sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Finetuning')
    parser.add_argument('--base_model_name', type=str, default='CohereForAI/aya-101')
    parser.add_argument(
        '--train_filename',
        type=str,
        default='/scratch/project_465001384/corpora/defgen_data/train_dbnary_de.tsv.gz',
    )
    parser.add_argument('--dev_filename', default='/scratch/project_465001384/corpora/defgen_data/dev_dbnary_de.tsv.gz')
    parser.add_argument('--lora_rank', type=int, default=256)
    parser.add_argument('--lora_alpha', type=int, default=512)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--finetuned_model_name', type=str, default='aya')
    parser.add_argument('--output_dir', type=str, default='/scratch/project_465001384/models/DefGen/aya_de')
    parser.add_argument('--max_seq_length', type=int, default=192)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_train_epochs', type=int, default=2)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--tag', default='')
    args = parser.parse_args()

    train(args)
