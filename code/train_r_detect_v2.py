import logging
import os
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from functools import partial

import datasets
import hydra
import pandas as pd
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

try:
    from r_detect.ai_dataset import AiDataset
    from r_detect.ai_loader import AiCollator, AiCollatorTrain, show_batch
    from utils.metric_utils import compute_metrics
    from utils.train_utils import AverageMeter, as_minutes, get_lr

except Exception as e:
    print(e)
    raise ImportError

logger = get_logger(__name__)


def run_evaluation(accelerator, model, valid_dl, valid_ids):
    model.eval()

    all_predictions = []
    all_truths = []

    progress_bar = tqdm(range(len(valid_dl)), disable=not accelerator.is_local_main_process)

    for step, batch in enumerate(valid_dl):
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.sigmoid(logits)
        predictions, references = accelerator.gather_for_metrics(
            (predictions, batch["labels"].to(torch.long).reshape(-1))
        )
        predictions, references = predictions.cpu().numpy().tolist(), references.cpu().numpy().tolist()

        all_predictions.extend(predictions)
        all_truths.extend(references)

        progress_bar.update(1)
    progress_bar.close()

    # compute metric
    eval_dict = compute_metrics(all_predictions, all_truths)

    result_df = pd.DataFrame()
    result_df["id"] = valid_ids
    result_df["predictions"] = all_predictions
    result_df["truths"] = all_truths

    oof_df = deepcopy(result_df)
    oof_df = oof_df.rename(columns={"predictions": "generated"})
    oof_df = oof_df[["id", "generated"]].copy()

    to_return = {
        "scores": eval_dict,
        "result_df": result_df,
        "oof_df": oof_df,
    }

    return to_return


@hydra.main(version_base=None, config_path="../conf/r_detect", config_name="conf_r_detect_t_lite")
def run_training(cfg):
    # ------- Accelerator ---------------------------------------------------------------#
    mixed_precision = "bf16" if cfg.training_args.get("bf16", False) else "no"
    
    if cfg.get("use_wandb", False):
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.training_args.gradient_accumulation_steps,
            log_with="wandb",
            mixed_precision=mixed_precision,
        )
        accelerator.init_trackers(
            cfg.get("wandb_project", "r_detect_v2"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.training_args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
        )

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    def print_line():
        prefix, unit, suffix = "#", "~~", "#"
        accelerator.print(prefix + unit * 50 + suffix)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # ------- Runtime Configs -----------------------------------------------------------#
    print_line()
    accelerator.print(f"setting seed: {cfg.get('seed', 42)}")
    set_seed(cfg.get("seed", 42))

    if accelerator.is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
    print_line()

    # ------- load data -----------------------------------------------------------------#
    print_line()
    data_dir = cfg.get("train_dataset", "datasets/final_prepared/final_train.csv").split("/")[0]
    data_dir = os.path.join(os.getcwd(), data_dir)
    
    train_path = cfg.train_dataset
    valid_path = cfg.valid_dataset
    
    if not os.path.exists(train_path):
        train_path = os.path.join(os.getcwd(), cfg.train_dataset)
    if not os.path.exists(valid_path):
        valid_path = os.path.join(os.getcwd(), cfg.valid_dataset)

    accelerator.print(f"Loading train from: {train_path}")
    accelerator.print(f"Loading valid from: {valid_path}")
    
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)

    # Rename is_generated to generated for consistency
    if "is_generated" in train_df.columns:
        train_df = train_df.rename(columns={"is_generated": "generated"})
    if "is_generated" in valid_df.columns:
        valid_df = valid_df.rename(columns={"is_generated": "generated"})

    # Add id column if missing
    if "id" not in train_df.columns:
        train_df["id"] = [f"train_{i}" for i in range(len(train_df))]
    if "id" not in valid_df.columns:
        valid_df["id"] = [f"val_{i}" for i in range(len(valid_df))]

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    accelerator.print(f"shape of train data: {train_df.shape}")
    accelerator.print(f"shape of validation data: {valid_df.shape}")
    accelerator.print(f"Train class distribution: {train_df['generated'].value_counts().to_dict()}")
    accelerator.print(f"Valid class distribution: {valid_df['generated'].value_counts().to_dict()}")

    with accelerator.main_process_first():
        dataset_creator = AiDataset(cfg)
        train_ds = dataset_creator.get_dataset(train_df)
        valid_ds = dataset_creator.get_dataset(valid_df)

    tokenizer = dataset_creator.tokenizer

    train_ds.set_format(
        type=None,
        columns=["id", "input_ids", "attention_mask", "generated"],
    )

    valid_ds = valid_ds.sort("input_length")
    valid_ds.set_format(
        type=None,
        columns=["id", "input_ids", "attention_mask", "generated"],
    )
    valid_ids = valid_df["id"]

    data_collator = AiCollator(tokenizer=tokenizer, pad_to_multiple_of=64)
    data_collator_train = AiCollatorTrain(tokenizer=tokenizer, pad_to_multiple_of=64, kwargs=dict(cfg=cfg))

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.training_args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator_train,
    )

    valid_dl = DataLoader(
        valid_ds,
        batch_size=cfg.training_args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    accelerator.print("data preparation done...")
    print_line()

    # --- model -------------------------------------------------------------------------#
    print_line()
    
    model_name = cfg.model_name
    accelerator.print(f"Loading model: {model_name}")
    
    tokenizer_name = cfg.get("tokenizer_name", model_name)
    accelerator.print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=cfg.num_labels,
        torch_dtype=torch.bfloat16 if cfg.training_args.bf16 else torch.float16,
        attn_implementation=cfg.training_args.get("attn_implementation", "flash_attention_2"),
        trust_remote_code=True,
    )

    # --- LoRA / DoRA configuration -----------------------------------------------------#
    lora_config = cfg.get("lora_config", {})
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_config.get("r", 32),
        lora_alpha=lora_config.get("lora_alpha", 64),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        bias=lora_config.get("bias", "none"),
        target_modules=lora_config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        use_rslora=lora_config.get("use_rslora", True),
        use_dora=lora_config.get("use_dora", True),
        modules_to_save=lora_config.get("modules_to_save", ["score"]),
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    accelerator.wait_for_everyone()

    # --- optimizer ---------------------------------------------------------------------#
    print_line()
    use_fused = (
        cfg.training_args.get("optim", "adamw_torch_fused") == "adamw_torch_fused"
        and next(model.parameters()).is_cuda
    )
    accelerator.print(f"Using fused AdamW: {use_fused}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training_args.learning_rate,
        weight_decay=cfg.training_args.weight_decay,
        fused=use_fused,
    )

    # ------- Prepare -------------------------------------------------------------------#
    model, optimizer, train_dl, valid_dl = accelerator.prepare(
        model, optimizer, train_dl, valid_dl
    )

    # ------- Scheduler -----------------------------------------------------------------#
    print_line()
    num_epochs = cfg.training_args.num_train_epochs
    grad_accumulation_steps = cfg.training_args.gradient_accumulation_steps
    warmup_ratio = cfg.training_args.warmup_ratio

    num_update_steps_per_epoch = len(train_dl) // grad_accumulation_steps
    num_training_steps = num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(warmup_ratio * num_training_steps)

    accelerator.print(f"# training updates per epoch: {num_update_steps_per_epoch}")
    accelerator.print(f"# training steps: {num_training_steps}")
    accelerator.print(f"# warmup steps: {num_warmup_steps}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # ------- training setup --------------------------------------------------------------#
    best_lb = -1.0
    patience_tracker = 0
    current_iteration = 0
    start_time = time.time()

    accelerator.wait_for_everyone()

    for epoch in range(num_epochs):
        if epoch != 0:
            progress_bar.close()

        progress_bar = tqdm(
            range(num_update_steps_per_epoch),
            disable=not accelerator.is_local_main_process,
        )
        loss_meter = AverageMeter()

        model.train()
        for step, batch in enumerate(train_dl):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        cfg.training_args.get("max_grad_norm", 1.0),
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                loss_meter.update(loss.item())

            if accelerator.sync_gradients:
                progress_bar.set_description(
                    f"STEP: {current_iteration+1:5}/{num_training_steps:5}. "
                    f"LR: {get_lr(optimizer):.6f}. "
                    f"Loss: {loss_meter.avg:.4f}. "
                )
                progress_bar.update(1)
                current_iteration += 1

                if cfg.get("use_wandb", False):
                    accelerator.log({"train_loss": round(loss_meter.avg, 5)}, step=current_iteration)
                    accelerator.log({"lr": get_lr(optimizer)}, step=current_iteration)

            # Evaluation
            if accelerator.sync_gradients and (current_iteration % cfg.training_args.eval_steps == 0):
                model.eval()
                eval_response = run_evaluation(accelerator, model, valid_dl, valid_ids)

                scores_dict = eval_response["scores"]
                result_df = eval_response["result_df"]
                oof_df = eval_response["oof_df"]

                lb = scores_dict["lb"]

                print_line()
                et = as_minutes(time.time() - start_time)
                accelerator.print(
                    f">>> Epoch {epoch+1} | Step {step} | Total Step {current_iteration} | Time: {et}"
                )
                print_line()
                accelerator.print(f">>> Current AUC = {round(lb, 4)}")
                print_line()

                is_best = False
                if lb >= best_lb:
                    best_lb = lb
                    is_best = True
                    patience_tracker = 0
                else:
                    patience_tracker += 1

                if is_best:
                    oof_df.to_csv(os.path.join(cfg.output_dir, "oof_df_best.csv"), index=False)
                    result_df.to_csv(os.path.join(cfg.output_dir, "result_df_best.csv"), index=False)
                else:
                    accelerator.print(f">>> patience reached {patience_tracker}/{cfg.training_args.get('patience', 3)}")

                oof_df.to_csv(os.path.join(cfg.output_dir, "oof_df_last.csv"), index=False)
                result_df.to_csv(os.path.join(cfg.output_dir, "result_df_last.csv"), index=False)

                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)

                unwrapped_model.save_pretrained(
                    os.path.join(cfg.output_dir, "last"),
                    state_dict=accelerator.get_state_dict(model),
                    save_function=accelerator.save,
                )

                if accelerator.is_main_process:
                    tokenizer.save_pretrained(os.path.join(cfg.output_dir, "last"))

                if is_best:
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(os.path.join(cfg.output_dir, "best"))
                    unwrapped_model.save_pretrained(
                        os.path.join(cfg.output_dir, "best"),
                        state_dict=accelerator.get_state_dict(model),
                        save_function=accelerator.save,
                    )

                if cfg.get("use_wandb", False):
                    accelerator.log({"auc": lb}, step=current_iteration)
                    accelerator.log({"best_auc": best_lb}, step=current_iteration)
                    for k, v in scores_dict.items():
                        accelerator.log({k: round(v, 4)}, step=current_iteration)

                model.train()
                torch.cuda.empty_cache()
                print_line()

                if patience_tracker >= cfg.training_args.get("patience", 3):
                    accelerator.print("stopping early")
                    model.eval()
                    accelerator.end_training()
                    return

    accelerator.end_training()


if __name__ == "__main__":
    run_training()
