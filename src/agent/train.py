"""
Main training agent. Using torch.compile and bfloat16 by default. Optionally (Q)LoRA.

"""

import logging
import os
from collections import deque

import bitsandbytes as bnb
import einops
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

import wandb
from src.agent.dataset import TorchRLDSInterleavedDataset
from src.agent.model_averaging import ModelAveraging
from src.utils.decorator import main_rank_only
from src.utils.metric import get_action_accuracy
from src.utils.monitor import (
    MainRankFilter,
    Timer,
    log_allocated_gpu_memory,
    log_execution_time,
)
from src.utils.optim import CosineAnnealingWarmupRestarts, get_num_params_in_billions

from src.model.vla import VLA

log = logging.getLogger(__name__)


class TrainVLAAgent: # changed to TrainVLAAgent
    def __init__(self, cfg):
        # device setup
        self.cfg = cfg
        self.gpu_id = cfg.gpu_id
        self.device = torch.device(f"cuda:{self.gpu_id}")
        self.multi_gpu = cfg.multi_gpu
        world_size = 1  # single gpu
        if self.multi_gpu:
            global_rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
            world_size = int(os.environ["WORLD_SIZE"])
            group_rank = int(os.environ["GROUP_RANK"])
            log.info(
                f"GPU local ID: {self.gpu_id}. Global rank: {global_rank}. Local rank: {local_rank}. Local world size: {local_world_size}. World size: {world_size}. Group rank: {group_rank}"
            )
            for i in range(torch.cuda.device_count()):
                log.info(
                    f"Local rank: {local_rank}, GPU UUID: {torch.cuda.get_device_properties(i).uuid}"
                )
        self.main_rank = not self.multi_gpu or global_rank == 0
        log.addFilter(MainRankFilter(main_rank=self.main_rank))


        self.debug = cfg.get("debug", False)
        self.inspect_data = cfg.get("inspect_data", False)
        self.save_model_freq = int(cfg.save_model_freq)
        self.save_model_start = int(cfg.get("save_model_start", 0))
        self.log_freq = cfg.log_freq
        self.log_dir = cfg.log_dir
        self.checkpoint_dir = os.path.join(self.log_dir, "checkpoint")
        os.makedirs(self.checkpoint_dir, exist_ok=True)




        # training params
        self.n_updates = int(cfg.n_updates)
        self.max_grad_norm = cfg.max_grad_norm
        self.use_amp = cfg.get("use_amp", True) # atomic mixed precision?
        self.dtype = torch.bfloat16 if cfg.get("use_bf16", True) else torch.float32
        self.use_torch_compile = cfg.get("use_torch_compile", True)


        # model
        assert not ((cfg.quantize or cfg.lora) and not cfg.load_pretrained_weights), (
            "Please load pretrained weights if quantizing VLM or using Lora."
        )
        if cfg.quantize and not cfg.lora:
            log.warning(
                "Quantizing VLM but not adding Lora weights, which means the VLM will be fully frozen!"
            )  # since the weights have requires_grad=False. However, we are not excluding the weights from the optimizer yet!
        self.model = VLA(cfg, use_ddp=self.multi_gpu)
        if cfg.resume_checkpoint_path:
            self.load_checkpoint(cfg.resume_checkpoint_path) # 
            # change other config params if loading a checkpoint
        elif cfg.load_pretrained_weights:
            self.model.load_pretrained_weights()


        # logging
        self.use_wandb = cfg.get("wandb", False) and (self.main_rank) and (cfg.use_wandb)
        if self.use_wandb:
            wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                name=cfg.wandb.run,
                config=OmegaConf.to_container(cfg, resolve=True),
                id=self.wandb_id if hasattr(self, "wandb_id") else None,
                resume="allow",  # not using resume_from
            )

        #self.model.tie_action_proprio_weights()
        #self.model.freeze_unused_weights()
        #if cfg.lora:
        #    self.model.freeze_non_lora_weights_in_vlm()
        self.model.to(self.dtype)
        self.model.to(self.device)
        if self.use_torch_compile: # having trouble getting to work - set to False nominally
            self.model = torch.compile(
                self.model,
                mode="default",  # "reduce-overhead" speeds up a lot and reduces VRAM usage a lot more, but causes nan loss on L40, maybe issue with cudagraphs or 8-bit optimizer; max-autotune works on H100s, takes a while to compile
                # backend="inductor", # default: inductor; cudagraphs
            )
        # modes: https://pytorch.org/docs/main/generated/torch.compile.html
        # backends: https://pytorch.org/docs/stable/torch.compiler.html
        log.info(f"Using cuda device: {self.device}, dtype: {self.dtype}")
        if self.multi_gpu:
            log.info(
                f"Using {local_world_size} GPUs in each of the {cfg.n_nodes} nodes"
            )
            import torch.distributed as dist
            from torch.nn.parallel import DistributedDataParallel as DDP

            self.model = DDP(
                self.model,
                device_ids=[local_rank],
                gradient_as_bucket_view=True, # groups gradients into a single buffer, reducing memory usage (prevents memory fragmentation)
                static_graph=False,  # attempts to reduce memory usage when the comp. graph doesn't change across iterations
            )
            model = self.model.module # use module to get the original model, otherwise it will be a DDP model
            dist.barrier() # wait for all ranks to be ready
        else:
            model = self.model # no need for module 
        log_allocated_gpu_memory(log, "loading model", self.gpu_id)

        # determine batch size and gradient accumulation steps
        self.grad_accumulation_steps = max(
            cfg.global_batch_size // cfg.per_device_batch_size // world_size, 1
        )
        actual_global_batch_size = (
            cfg.per_device_batch_size * self.grad_accumulation_steps * world_size
        )




        # dataloader --- spawn one for each rank, num_workers=0
        print("Setting up dataloader")
        print("CHANGE COLLATE FUNCTION BACK TO DEFAULT TO TRAIN HOW OPENPIZERO DOES IT")
        print("MAKE INTERWEAVE FUNCTION TO GET BACK TO HOW OPENPIZERO DOES IT")
        self.train_dataloader = DataLoader(
            TorchRLDSInterleavedDataset(cfg.data.train, train=True).dataset,
            batch_size=cfg.per_device_batch_size,
            pin_memory=True,
            collate_fn=self.custom_collate_fn # delete this line to use default collate function
        )
        self.run_eval = cfg.data.get("val", False)
        if self.run_eval:
            cfg_data_val = OmegaConf.merge(cfg.data.train, cfg.data.val)
            self.val_dataiterator = iter(
                DataLoader(
                    TorchRLDSInterleavedDataset(cfg_data_val, train=False).dataset,
                    batch_size=cfg.per_device_batch_size,
                    pin_memory=True,
                )
            )
            self.eval_thresholds = cfg.eval_thresholds
            self.eval_freq = cfg.eval_freq
            self.per_device_num_eval_batch = (
                cfg.eval_size // cfg.per_device_batch_size // world_size
            )
        log.info(f"Total number of gradient updates: {self.n_updates}")
        log.info(f"Global batch size: {actual_global_batch_size}")
        log.info(f"Per device batch size: {cfg.per_device_batch_size}")
        log.info(f"Gradient accumulation steps: {self.grad_accumulation_steps}")

        # optimizer
        self.trained_parameters = model.trainable_parameters
        self.optimizer = bnb.optim.AdamW8bit(
            model.trainable_parameters,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.lr_scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=cfg.lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.lr,
            min_lr=cfg.lr_scheduler.min_lr,
            warmup_steps=cfg.lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        log.info(
            f"Number of trained parameters: {get_num_params_in_billions(self.optimizer):.3f}B"
        )

        # load optimizer if resuming
        if cfg.resume_checkpoint_path:
            self.load_optimizer(cfg.resume_checkpoint_path)

        ########### Input processing ###########
        self.tokenizer = model.text_tokenizer

        print("Done initializing agent")

    def run(self):
        timer = Timer()
        cnt_batch = 0 if not hasattr(self, "cnt_batch") else self.cnt_batch
        cnt_update = (
            0 if not hasattr(self, "cnt_update") else self.cnt_update
        )  # resume training if loaded checkpoint
        loss_deque = deque(maxlen=self.grad_accumulation_steps)
        new_eval_from_last_log = False

        # deal with the various model.module
        model_meta = self.model
        if self.multi_gpu:
            import torch.distributed as dist
            model = self.model.module
        else:
            model = self.model
        model_meta.train()

        # Set up model averaging
        self.model_averaging = ModelAveraging(model, self.cfg, self.device)

        def preprocess_batch(batch):
            #TODO(ajhancock): add history
            images = batch["observation"]["image_primary"] # should b
            images = einops.rearrange(
                images, "B T H W C -> B (T C) H W"
            ).float() # move to float for vision encoders  # remove cond_steps dimension
            #proprios = batch["observation"]["proprio"]
            #actions = batch["action"].squeeze(1)  # remove the time dimension
            actions = batch["action"].to("cuda")
            # assume horizon and time dimension are the same, 1
            actions = einops.rearrange(actions, "B 1 1 D -> B D") 

            texts = [
                text.decode("utf-8") for text in batch["task"]["language_instruction"]
            ]
            texts = np.array(texts)
            tokenized_texts = self.tokenizer(texts)
            #model_inputs = self.processor(text=texts, images=images)

            # specific to how VLA is defined
            inputs = {
                "image": images.to(self.dtype),
                #"texts": texts,
                "text_tokens": tokenized_texts,
                "actions": actions.to(self.dtype),
            }
           
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            return inputs
        
        def preprocess_trajectory(batch):
            # now, each batch is an entire trajectory (of different lengths)
            #TODO: pad, but for now lets just organize

            pug = 2

        while True:
            for batch in self.train_dataloader:
                """
                batch: dict with keys 'observation', 'task', 'action', 'dataset_name', 'action_pad_mask'
                observation: 'image_primary' (torch.Size([bsz, 1, H, W, 3], uint8), 'image_wrist', 'timestep' (torch.Size([bsz, 1])), 'pad_mask_dict', 'timestep_pad_mask', 'task_completed' (torch.Size([bsz, window, 4]), 'proprio' (torch.Size([bsz, window, proprio_dim])
                task: 'language_instruction', 'pad_mask_dict', 'image_primary', 'image_wrist', 'timestep' (torch.Size([bsz]))
                action (torch.Size([bsz, window, horizon, action_dim], float32)
                action_pad_mask (torch.Size([bsz, window, horizon, action_dim]))
                """

               
                if self.inspect_data:
                    #breakpoint()
                    print("Inspecting data")
                    
                    # Save batch as pickle file
                    # save in log directory
                    save_dir = os.path.join(self.log_dir, "batch_data")
                    import pickle
                    with open(f"{save_dir}.pkl", "wb") as f:
                        pickle.dump(batch, f)
                    exit()
                    
                inputs = preprocess_batch(batch) 

                # debugging
                if self.debug and cnt_batch == 0:
                    images = batch["observation"]["image_primary"] # should be fine
                    proprios = batch["observation"]["proprio"]
                    actions = batch["action"]
                    actions = einops.rearrange(actions, "B T H D -> B D")
                    texts = [
                        text.decode("utf-8")
                        for text in batch["task"]["language_instruction"]
                    ]
                    log.info(f"{self.gpu_id} device {self.device}")
                    log.info(f"{self.gpu_id} texts {texts}")
                    log.info(f"{self.gpu_id} images {images.shape}")
                    log.info(
                        f"{self.gpu_id} actions {actions.shape} {actions.mean()} {actions.std()} {actions.max()} {actions.min()}"
                    )
                    log.info(
                        f"{self.gpu_id} proprios {proprios.shape} {proprios.mean()} {proprios.std()} {proprios.max()} {proprios.min()}"
                    )

                    # Save an image for debugging
                    #image = images[0, 0].clone().to("cpu")
                    #image = Image.fromarray(image.numpy())
                    #image.save(os.path.join(self.log_dir, f"image_{self.gpu_id}.png"))


                # make sure only syncing when taking gradient steps
                if (cnt_batch + 1) % self.grad_accumulation_steps != 0:
                    with model_meta.no_sync():
                        with torch.autocast(
                            device_type="cuda", dtype=self.dtype, enabled=self.use_amp
                        ):
                            loss = model_meta(**inputs, mode="train")
                        if self.debug:
                            log_allocated_gpu_memory(log, f"forward batch {cnt_batch}")
                        normalized_loss = loss / self.grad_accumulation_steps
                        normalized_loss.backward()
                else:
                    with torch.autocast(
                        device_type="cuda", dtype=self.dtype, enabled=self.use_amp
                    ):
                        loss = model_meta(**inputs, mode="train")
                    if self.debug:
                        log_allocated_gpu_memory(log, f"forward batch {cnt_batch}")
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()  # gradients synced

                    # step
                    torch.nn.utils.clip_grad_norm_(
                        self.trained_parameters,
                        max_norm=self.max_grad_norm,
                    )

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    if self.debug:
                        log_allocated_gpu_memory(
                            log, f"optimizer step batch {cnt_batch}"
                        )
                    self.optimizer.zero_grad(set_to_none=True)
                    cnt_update += 1

                    # initialize ema/swa
                    self.model_averaging.maybe_initialize(cnt_update)

                    # update ema/swa
                    self.model_averaging.maybe_update(cnt_update)

                    # save model at the end of update, models just synced
                    if (
                        cnt_update % self.save_model_freq == 0
                        and cnt_update > self.save_model_start
                    ) or cnt_update == self.n_updates:
                        self.save_training(
                            cnt_update, cnt_batch, main_rank=self.main_rank
                        )
                        dist.barrier()

                # aggregate loss
                if self.multi_gpu:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss_deque.append(loss.item() / dist.get_world_size())
                else:
                    loss_deque.append(loss.item())

                # validation with action accuracy
                if self.run_eval and (cnt_batch + 1) % self.eval_freq == 0:
                    log.info(
                        f"Running evaluation for {self.per_device_num_eval_batch} batches..."
                    )
                    new_eval_from_last_log = True
                    model_meta.eval()
                    model_eval = self.model_averaging.get_model_module()
                    eval_accuracy = torch.zeros(
                        len(self.eval_thresholds), device=self.device
                    )
                    eval_l1_loss = torch.tensor(0.0, device=self.device)
                    with torch.inference_mode():
                        for _ in range(self.per_device_num_eval_batch):
                            batch_eval = next(self.val_dataiterator)
                            inputs = preprocess_batch(
                                batch_eval
                            )
                            gt_actions = inputs.pop("actions")
                            preds = model_eval(**inputs, mode="test")
                            eval_accuracy += get_action_accuracy(
                                gt_actions, preds, self.eval_thresholds
                            )
                            eval_l1_loss += torch.nn.functional.l1_loss(
                                preds, gt_actions
                            )
                    model_meta.train()

                    # get stats
                    eval_accuracy = eval_accuracy / self.per_device_num_eval_batch
                    eval_l1_loss = eval_l1_loss / self.per_device_num_eval_batch
                    if self.multi_gpu:
                        dist.all_reduce(eval_accuracy, op=dist.ReduceOp.SUM)
                        dist.all_reduce(eval_l1_loss, op=dist.ReduceOp.SUM)
                        eval_accuracy /= dist.get_world_size()
                        eval_l1_loss /= dist.get_world_size()
                    log_msg = f"Eval | l1 Loss: {eval_l1_loss.item():.3f} | "
                    log_msg += " | ".join(
                        [
                            f"acc thres {threshold}: {accuracy.item():.3f}"
                            for threshold, accuracy in zip(
                                self.eval_thresholds, eval_accuracy
                            )
                        ]
                    )
                    log.info(log_msg)

                # log loss
                if cnt_batch % self.log_freq == 0:
                    loss_metric = np.mean(loss_deque)
                    peak_vram = torch.cuda.max_memory_reserved(self.gpu_id) / (1024**3)
                    log_msg = f"Batch {cnt_batch} Update {cnt_update}: t {timer():8.4f} | vram {peak_vram:6.3f} | loss {loss_metric:6.4f} | lr {self.optimizer.param_groups[0]['lr']:10.8f}"
                    log.info(log_msg)
                    if self.use_wandb:
                        wandb_metrics = {
                            "loss - train": loss_metric,
                            "gradient steps": cnt_update,
                            "lr": self.optimizer.param_groups[0]["lr"],
                        }
                        if new_eval_from_last_log:
                            wandb_metrics.update(
                                {
                                    f"eval acc - thres {threshold}": accuracy.item()
                                    for threshold, accuracy in zip(
                                        self.eval_thresholds, eval_accuracy
                                    )
                                }
                            )
                            wandb_metrics["eval l1 loss"] = eval_l1_loss.item()
                            new_eval_from_last_log = False
                        wandb.log(wandb_metrics, step=cnt_batch, commit=True)

                # count
                cnt_batch += 1
                if cnt_update >= self.n_updates:
                    return

    @main_rank_only
    @log_execution_time(log)
    def save_training(self, cnt_update: int, cnt_batch: int, main_rank: bool):
        avg_state = self.model_averaging.state_dict()
        model_type = avg_state.get("model_type", "normal")
        n_averaged = avg_state.get("n_averaged", 1)
        if avg_state:
            weights = avg_state["state_dict"]
        elif self.multi_gpu:
            weights = self.model.module.state_dict()
        else:
            weights = self.model.state_dict()
        data = {
            "cnt_update": cnt_update,
            "cnt_batch": cnt_batch,
            "model": weights,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "wandb_id": wandb.run.id if self.use_wandb else None,
            "n_averaged": n_averaged,
        }
        savepath = os.path.join(self.checkpoint_dir, f"step{cnt_update}.pt")
        torch.save(data, savepath)
        checkpoint_size_in_gb = os.path.getsize(savepath) / (1024**3)
        log.info(
            f"Saved model to {savepath}, size: {checkpoint_size_in_gb:.2f} GB, type: {model_type}, averaged: {n_averaged}"
        )

    @log_execution_time(log)
    def load_checkpoint(self, path: str):
        """load to cpu first, then move to gpu"""
        data = torch.load(path, weights_only=True, map_location="cpu")
        self.cnt_update = data["cnt_update"]
        self.cnt_batch = data["cnt_batch"]
        self.wandb_id = data["wandb_id"]
        data["model"] = {
            k.replace("_orig_mod.", ""): v for k, v in data["model"].items()
        }  # remove "_orig_mod." prefix if saved model was compiled
        self.model.load_state_dict(data["model"], strict=True)
        log.info(
            f"Loaded model from {path} at update {self.cnt_update} batch {self.cnt_batch}"
        )

    @log_execution_time(log)
    def load_optimizer(self, path: str):
        """load to cpu first, then move to gpu"""
        from src.utils.optim import optimizer_to

        data = torch.load(path, weights_only=True, map_location="cpu")
        self.optimizer.load_state_dict(data["optimizer"])
        optimizer_to(self.optimizer, self.device)
        self.lr_scheduler.load_state_dict(data["lr_scheduler"])
        log.info(f"Loaded optimizer and scheduler states from {path}")


    def custom_collate_fn(self, batch):
        """
        Custom collate function to handle variable-length trajectories and complex data types.
        
        Args:
            batch (list): List of individual trajectory samples
        
        Returns:
            dict: Collated batch with padded trajectories
        """
        # Identify keys in the batch
        keys = batch[0].keys()
        
        # Prepare output dictionary
        collated_batch = {}
        
        for key in keys:
            # Collect all values for this key
            values = [item[key] for item in batch]
            
            # Handle different types of data
            if isinstance(values[0], dict):
                # If the value is a dictionary, keep as list
                collated_batch[key] = values
            
            elif isinstance(values[0], (np.ndarray, torch.Tensor)):
                # Tensor/Array handling
                try:
                    # Try to stack numeric arrays
                    if all(isinstance(v, (int, float, np.number)) for v in np.concatenate(values)):
                        # Find the maximum length
                        max_len = max(v.shape[0] for v in values)
                        
                        # Pad tensors to the maximum length
                        padded_values = []
                        mask_values = []
                        for v in values:
                            if v.shape[0] < max_len:
                                # Create padding
                                if len(v.shape) == 1:
                                    # 1D tensor padding
                                    pad_dtype = v.dtype if hasattr(v, 'dtype') else type(v[0])
                                    pad = np.zeros((max_len - v.shape[0],) + v.shape[1:], dtype=pad_dtype)
                                    padded_v = np.concatenate([v, pad])
                                    mask_v = np.concatenate([np.ones(v.shape[0], dtype=bool), 
                                                            np.zeros(max_len - v.shape[0], dtype=bool)])
                                else:
                                    # Multi-dimensional tensor padding
                                    pad_dtype = v.dtype if hasattr(v, 'dtype') else type(v[0][0])
                                    pad_shape = (max_len - v.shape[0],) + v.shape[1:]
                                    pad = np.zeros(pad_shape, dtype=pad_dtype)
                                    padded_v = np.concatenate([v, pad], axis=0)
                                    mask_v = np.concatenate([np.ones(v.shape[0], dtype=bool), 
                                                            np.zeros(max_len - v.shape[0], dtype=bool)])
                            else:
                                padded_v = v
                                mask_v = np.ones(max_len, dtype=bool)
                            
                            padded_values.append(padded_v)
                            mask_values.append(mask_v)
                        
                        # Convert to tensor
                        collated_batch[key] = torch.tensor(np.stack(padded_values))
                        
                        # Add a mask to indicate valid entries
                        collated_batch[f'{key}_mask'] = torch.tensor(np.stack(mask_values))
                    else:
                        # If not all numeric, keep as list
                        collated_batch[key] = values
                except Exception:
                    # Fallback to keeping original values if stacking fails
                    collated_batch[key] = values
            
            elif isinstance(values[0], (int, float, str)):
                # Simple types like numbers or strings
                try:
                    collated_batch[key] = torch.tensor(values)
                except Exception:
                    # If conversion fails, keep as list
                    collated_batch[key] = values
            
            else:
                # For any other type, keep the original list
                collated_batch[key] = values
        
        return collated_batch
# Ensure it's importable
__all__ = ["TrainVLAAgent"]