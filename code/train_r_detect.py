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
from peft import (LoraConfig, TaskType, get_peft_model)
# bitsandbytes is not imported to avoid CUDA setup issues
# We use standard PyTorch optimizers instead
BITSANDBYTES_AVAILABLE = False
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoModelForSequenceClassification,
                          BitsAndBytesConfig, get_cosine_schedule_with_warmup)

try:
    from r_detect.ai_dataset import AiDataset
    from r_detect.ai_loader import AiCollator, AiCollatorTrain, show_batch
    from r_detect.ai_model import (LlamaForDetectAI, MistralForDetectAI,
                                   PhiForDetectAI)
    from r_detect.ai_optimizer import get_optimizer
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
        predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"].to(torch.long).reshape(-1)))
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


@hydra.main(version_base=None, config_path="../conf/r_detect", config_name="conf_r_detect")
def run_training(cfg):
    # ------- Accelerator ---------------------------------------------------------------#
    if cfg.use_wandb:
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.train_params.gradient_accumulation_steps,
            log_with="wandb",
            # mixed_precision='fp16',
        )

        accelerator.init_trackers(
            cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.train_params.gradient_accumulation_steps,
            # mixed_precision='fp16',
        )

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # print_line = partial(print_line, accelerator)

    def print_line():
        prefix, unit, suffix = "#", "~~", "#"
        accelerator.print(prefix + unit*50 + suffix)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # ------- Runtime Configs -----------------------------------------------------------#
    print_line()
    accelerator.print(f"setting seed: {cfg.seed}")
    set_seed(cfg.seed)

    if accelerator.is_main_process:
        os.makedirs(cfg.outputs.model_dir, exist_ok=True)
    print_line()

    # ------- load data -----------------------------------------------------------------#
    print_line()
    data_dir = cfg.input_data_dir
    
    # Check for external dataset (for mix_v16, mix_v26 configs)
    external_data_dir = cfg.get('external_data_dir', None)

    # First check for final_train.csv and final_valid.csv (for final_dataset)
    final_train_path = os.path.join(data_dir, "final_train.csv")
    final_valid_path = os.path.join(data_dir, "final_valid.csv")
    
    if os.path.exists(final_train_path) and os.path.exists(final_valid_path):
        accelerator.print("Using final_train.csv and final_valid.csv for training")
        try:
            train_df = pd.read_csv(final_train_path)
            valid_df = pd.read_csv(final_valid_path)
            
            # Rename is_generated to generated for consistency
            if 'is_generated' in train_df.columns:
                train_df = train_df.rename(columns={'is_generated': 'generated'})
            if 'is_generated' in valid_df.columns:
                valid_df = valid_df.rename(columns={'is_generated': 'generated'})
            
            # Add id column if missing
            if 'id' not in train_df.columns:
                train_df['id'] = [f"final_train_{i}" for i in range(len(train_df))]
            if 'id' not in valid_df.columns:
                valid_df['id'] = [f"final_valid_{i}" for i in range(len(valid_df))]
            
            accelerator.print(f"Loaded final_train.csv: {train_df.shape}")
            accelerator.print(f"Loaded final_valid.csv: {valid_df.shape}")
                
        except Exception as e:
            accelerator.print(f"Error loading final CSVs: {e}")
            raise e
        
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        
        accelerator.print(f"shape of train data: {train_df.shape}")
        accelerator.print(f"{train_df.head()}")
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
            columns=['id', 'input_ids', 'attention_mask', 'generated']
        )
        
        valid_ds = valid_ds.sort("input_length")
        valid_ds.set_format(
            type=None,
            columns=['id', 'input_ids', 'attention_mask', 'generated']
        )
        valid_ids = valid_df["id"]
        
        data_collator = AiCollator(tokenizer=tokenizer, pad_to_multiple_of=64)
        data_collator_train = AiCollatorTrain(tokenizer=tokenizer, pad_to_multiple_of=64, kwargs=dict(cfg=cfg))
        
        train_dl = DataLoader(train_ds, batch_size=cfg.train_params.per_device_train_batch_size, shuffle=True, collate_fn=data_collator_train)
        valid_dl = DataLoader(valid_ds, batch_size=cfg.train_params.per_device_eval_batch_size, shuffle=False, collate_fn=data_collator)
        
        accelerator.print("data preparation done...")
        print_line()
        
        # Continue to model creation...
        goto_model_creation = True
    else:
        goto_model_creation = False
        # Try to load train_essays.csv as fallback
        try:
            essay_df = pd.read_csv(os.path.join(data_dir, "train_essays.csv"))
        except Exception as e:
            essay_df = pd.read_parquet(os.path.join(data_dir, "train_essays.parquet"))

        essay_df = essay_df[~essay_df['text'].isna()].copy()
        essay_df = essay_df.reset_index(drop=True)
        
        # Load external dataset if specified (for mix models)
        if external_data_dir is not None and os.path.exists(external_data_dir):
            accelerator.print(f"Loading external dataset from: {external_data_dir}")
            try:
                external_df = pd.read_csv(os.path.join(external_data_dir, "train_essays.csv"))
            except Exception as e:
                try:
                    external_df = pd.read_parquet(os.path.join(external_data_dir, "train_essays.parquet"))
                except Exception as e2:
                    try:
                        external_df = pd.read_csv(os.path.join(external_data_dir, "train.csv"))
                    except Exception as e3:
                        external_df = pd.read_parquet(os.path.join(external_data_dir, "train.parquet"))
            
            external_df = external_df[~external_df['text'].isna()].copy()
            external_df = external_df.reset_index(drop=True)
            accelerator.print(f"External dataset shape: {external_df.shape}")
            
            # Combine datasets
            essay_df = pd.concat([essay_df, external_df], ignore_index=True)
            essay_df = essay_df.reset_index(drop=True)
            accelerator.print(f"Combined dataset shape: {essay_df.shape}")

        # Use detection_train.csv and detection_val.csv if available
        train_csv_path = os.path.join(data_dir, "detection_train.csv")
        val_csv_path = os.path.join(data_dir, "detection_val.csv")
        
        if os.path.exists(train_csv_path) and os.path.exists(val_csv_path):
            accelerator.print("Using detection_train.csv and detection_val.csv for training")
            try:
                train_df = pd.read_csv(train_csv_path)
                valid_df = pd.read_csv(val_csv_path)
                
                # Rename is_generated to generated for consistency
                if 'is_generated' in train_df.columns:
                    train_df = train_df.rename(columns={'is_generated': 'generated'})
                if 'is_generated' in valid_df.columns:
                    valid_df = valid_df.rename(columns={'is_generated': 'generated'})
                
                # Add id column if missing
                if 'id' not in train_df.columns:
                    train_df['id'] = [f"train_{i}" for i in range(len(train_df))]
                if 'id' not in valid_df.columns:
                    valid_df['id'] = [f"val_{i}" for i in range(len(valid_df))]
                    
            except Exception as e:
                accelerator.print(f"Error loading detection CSVs: {e}")
                accelerator.print("Falling back to train_essays.csv with custom split")
                # Fallback to custom split
                n_pos = essay_df['generated'].sum()
                n_neg = len(essay_df) - n_pos
                n_pos_valid = min(2, n_pos) if n_pos >= 2 else n_pos
                n_neg_valid = max(int(n_neg * 0.05), 50)
                
                pos_df = essay_df[essay_df['generated'] == 1]
                neg_df = essay_df[essay_df['generated'] == 0]
                
                pos_train, pos_valid = train_test_split(
                    pos_df, test_size=n_pos_valid, random_state=cfg.seed
                )
                neg_train, neg_valid = train_test_split(
                    neg_df, test_size=n_neg_valid, random_state=cfg.seed
                )
                
                train_df = pd.concat([pos_train, neg_train], ignore_index=True).sample(frac=1, random_state=cfg.seed).reset_index(drop=True)
                valid_df = pd.concat([pos_valid, neg_valid], ignore_index=True).sample(frac=1, random_state=cfg.seed).reset_index(drop=True)
        else:
            # Fallback to custom split from train_essays.csv
            accelerator.print("Using train_essays.csv with custom split")
            n_pos = essay_df['generated'].sum()
            n_neg = len(essay_df) - n_pos
            n_pos_valid = min(2, n_pos) if n_pos >= 2 else n_pos
            n_neg_valid = max(int(n_neg * 0.05), 50)
            
            pos_df = essay_df[essay_df['generated'] == 1]
            neg_df = essay_df[essay_df['generated'] == 0]
            
            pos_train, pos_valid = train_test_split(
                pos_df, test_size=n_pos_valid, random_state=cfg.seed
            )
            neg_train, neg_valid = train_test_split(
                neg_df, test_size=n_neg_valid, random_state=cfg.seed
            )
            
            train_df = pd.concat([pos_train, neg_train], ignore_index=True).sample(frac=1, random_state=cfg.seed).reset_index(drop=True)
            valid_df = pd.concat([pos_valid, neg_valid], ignore_index=True).sample(frac=1, random_state=cfg.seed).reset_index(drop=True)

        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)

        accelerator.print(f"shape of train data: {train_df.shape}")
        accelerator.print(f"{train_df.head()}")
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
            columns=[
                'id',
                'input_ids',
                'attention_mask',
                'generated'
            ]
        )

        valid_ds = valid_ds.sort("input_length")

        valid_ds.set_format(
            type=None,
            columns=[
                'id',
                'input_ids',
                'attention_mask',
                'generated'
            ]
        )
        valid_ids = valid_df["id"]  # .tolist()

        data_collator = AiCollator(
            tokenizer=tokenizer,
            pad_to_multiple_of=64
        )
        data_collator_train = AiCollatorTrain(
            tokenizer=tokenizer,
            pad_to_multiple_of=64,
            kwargs=dict(cfg=cfg)
        )

        train_dl = DataLoader(
            train_ds,
            batch_size=cfg.train_params.per_device_train_batch_size,
            shuffle=True,
            collate_fn=data_collator_train,
        )

        valid_dl = DataLoader(
            valid_ds,
            batch_size=cfg.train_params.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )

        accelerator.print("data preparation done...")
        print_line()

        # --- show batch -------------------------------------------------------------------#
        print_line()

        for b in train_dl:
            break
        show_batch(b, tokenizer, task='training', print_fn=accelerator.print)

        print_line()

        for b in valid_dl:
            break
        show_batch(b, tokenizer, task='training', print_fn=accelerator.print)

    # --- model -------------------------------------------------------------------------#
    print_line()
    
    # Check if 4-bit quantization should be used
    use_4bit = BITSANDBYTES_AVAILABLE and cfg.model.get('use_4bit', True)
    
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        accelerator.print("Using 4-bit quantization (QLoRA)")
    else:
        bnb_config = None
        accelerator.print("WARNING: Using standard training without 4-bit quantization (bitsandbytes not available or disabled)")
    
    if 'solar' in cfg.model.backbone_path.lower():
        base_model = LlamaForDetectAI.from_pretrained(
            cfg.model.backbone_path,
            num_labels=cfg.model.num_labels,  # 2
            quantization_config=bnb_config,
        )
    elif 'phi' in cfg.model.backbone_path.lower():
        base_model = PhiForDetectAI.from_pretrained(
            cfg.model.backbone_path,
            num_labels=cfg.model.num_labels,  # 2
            quantization_config=bnb_config,
            trust_remote_code=True,  # IMP
        )
    else:
        base_model = MistralForDetectAI.from_pretrained(
            cfg.model.backbone_path,
            num_labels=cfg.model.num_labels,  # 2
            quantization_config=bnb_config,
        )
        # base_model.peft_config = dict()

    base_model.config.pretraining_tp = 1
    # base_model.config.pad_token_id = tokenizer.pad_token_id

    # # base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    # for param in base_model.parameters():
    #     if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
    #         param.data = param.data.to(torch.float32)

    # lora ---
    peft_config = LoraConfig(
        r=cfg.model.lora.r,
        lora_alpha=cfg.model.lora.lora_alpha,
        lora_dropout=cfg.model.lora.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        target_modules=cfg_dict["model"]["lora"]["target_modules"],
        modules_to_save=cfg_dict["model"]["lora"]["modules_to_save"],
    )

    model = get_peft_model(base_model, peft_config)
    print(model.device)
    model.print_trainable_parameters()
    accelerator.wait_for_everyone()

    # --- optimizer ---------------------------------------------------------------------#
    print_line()
    optimizer = get_optimizer(cfg, model, print_fn=accelerator.print)

    # ------- Prepare -------------------------------------------------------------------#

    model, optimizer, train_dl, valid_dl = accelerator.prepare(
        model, optimizer, train_dl, valid_dl
    )

    # ------- Scheduler -----------------------------------------------------------------#
    print_line()
    num_epochs = cfg.train_params.num_train_epochs
    grad_accumulation_steps = cfg.train_params.gradient_accumulation_steps
    warmup_pct = cfg.train_params.warmup_pct

    num_update_steps_per_epoch = len(train_dl)//grad_accumulation_steps
    num_training_steps = num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(warmup_pct*num_training_steps)

    accelerator.print(f"# training updates per epoch: {num_update_steps_per_epoch}")
    accelerator.print(f"# training steps: {num_training_steps}")
    accelerator.print(f"# warmup steps: {num_warmup_steps}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # scheduler = accelerator.prepare(scheduler)

    # ------- training setup --------------------------------------------------------------#
    best_lb = -1.
    save_trigger = cfg.train_params.save_trigger

    patience_tracker = 0
    current_iteration = 0

    # ------- training  --------------------------------------------------------------------#
    start_time = time.time()
    accelerator.wait_for_everyone()

    for epoch in range(num_epochs):
        # close and reset progress bar
        if epoch != 0:
            progress_bar.close()

        progress_bar = tqdm(range(num_update_steps_per_epoch), disable=not accelerator.is_local_main_process)
        loss_meter = AverageMeter()

        # Training ------
        model.train()
        for step, batch in enumerate(train_dl):
            with accelerator.accumulate(model):  # gives sync vs no sync context manager
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    # Q: why need this check?
                    # A: gradient_state.sync_gradients check is NOT performed inside clip_grad_norm_
                    accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.max_grad_norm)

                    optimizer.step()  # gradient_state.sync_gradients check is performed inside optimizer.step
                    scheduler.step()
                    optimizer.zero_grad()

                # check if loss.item() is okay for TPU
                # happening on all processes - values of loss meter in different processes are different
                loss_meter.update(loss.item())  # tracks loss in each batch, no accumulation

            if accelerator.sync_gradients:
                progress_bar.set_description(
                    f"STEP: {current_iteration+1:5}/{num_training_steps:5}. "
                    f"LR: {get_lr(optimizer):.4f}. "
                    f"Loss: {loss_meter.avg:.4f}. "
                )

                progress_bar.update(1)
                current_iteration += 1

                if cfg.use_wandb:
                    accelerator.log({"train_loss": round(loss_meter.avg, 5)}, step=current_iteration)  # only on main process
                    accelerator.log({"lr": get_lr(optimizer)}, step=current_iteration)

            # >--------------------------------------------------|
            # >-- evaluation ------------------------------------|
            # >--------------------------------------------------|

            if (accelerator.sync_gradients) & (current_iteration % cfg.train_params.eval_frequency == 0):
                # set model in eval mode
                model.eval()
                eval_response = run_evaluation(accelerator, model, valid_dl, valid_ids)

                scores_dict = eval_response["scores"]
                result_df = eval_response["result_df"]
                oof_df = eval_response["oof_df"]

                lb = scores_dict["lb"]

                print_line()
                et = as_minutes(time.time()-start_time)
                accelerator.print(
                    f">>> Epoch {epoch+1} | Step {step} | Total Step {current_iteration} | Time: {et}"
                )
                print_line()
                accelerator.print(f">>> Current LB (AUC) = {round(lb, 4)}")

                print_line()

                is_best = False
                if lb >= best_lb:
                    best_lb = lb
                    is_best = True
                    patience_tracker = 0

                    # -----
                    best_dict = dict()
                    for k, v in scores_dict.items():
                        best_dict[f"{k}_at_best"] = v
                else:
                    patience_tracker += 1

                if is_best:  # do in main process
                    oof_df.to_csv(os.path.join(cfg.outputs.model_dir, f"oof_df_best.csv"), index=False)
                    result_df.to_csv(os.path.join(cfg.outputs.model_dir, f"result_df_best.csv"), index=False)
                else:
                    accelerator.print(f">>> patience reached {patience_tracker}/{cfg.train_params.patience}")
                    accelerator.print(f">>> current best score: {round(best_lb, 4)}")

                oof_df.to_csv(os.path.join(cfg.outputs.model_dir, f"oof_df_last.csv"), index=False)
                result_df.to_csv(os.path.join(cfg.outputs.model_dir, f"result_df_last.csv"), index=False)

                # saving -----
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)

                # # debug --
                # selected_adapters = list(unwrapped_model.peft_config.keys())
                # accelerator.print(f"selected adapters: {selected_adapters}")
                # for adapter_name in selected_adapters:
                #     peft_config = unwrapped_model.peft_config[adapter_name]
                #     peft_config = asdict(peft_config)
                #     accelerator.print(f"adapter: {adapter_name}")
                #     accelerator.print(peft_config)
                #     for k, v in peft_config.items():
                #         accelerator.print(f"{k}: {v} ({type(v)})")
                # # ------
                unwrapped_model.save_pretrained(
                    f"{cfg.outputs.model_dir}/last",
                    state_dict=accelerator.get_state_dict(model),
                    save_function=accelerator.save,
                )

                if accelerator.is_main_process:
                    tokenizer.save_pretrained(f"{cfg.outputs.model_dir}/last")

                if best_lb > save_trigger:
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(f"{cfg.outputs.model_dir}/best")
                    unwrapped_model.save_pretrained(
                        f"{cfg.outputs.model_dir}/best",
                        state_dict=accelerator.get_state_dict(model),
                        save_function=accelerator.save,
                    )
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(f"{cfg.outputs.model_dir}/best")

                # logging ----
                if cfg.use_wandb:
                    accelerator.log({"lb": lb}, step=current_iteration)
                    accelerator.log({"best_lb": best_lb}, step=current_iteration)

                    # -- log scores dict
                    for k, v in scores_dict.items():
                        accelerator.log({k: round(v, 4)}, step=current_iteration)

                    # --- log best scores dict
                    for k, v in best_dict.items():
                        accelerator.log({k: round(v, 4)}, step=current_iteration)

                # -- post eval
                model.train()
                torch.cuda.empty_cache()
                print_line()

                # early stopping ----
                if patience_tracker >= cfg.train_params.patience:
                    print("stopping early")
                    model.eval()
                    accelerator.end_training()
                    return

    # --- end training
    accelerator.end_training()


if __name__ == "__main__":
    run_training()
