import torch
import torch.nn as nn
import torch.utils.data
from packaging import version
from typing import Dict, List, Optional, Union, Tuple, Callable, Any
from dataclasses import dataclass
import textwrap
import copy
import os
import transformers
from collections import defaultdict
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    AutoModelForCausalLM,
    AutoProcessor,
    Trainer,
    TrainerCallback,
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    GenerationConfig,
    is_wandb_available,
)
from transformers.trainer_utils import EvalPrediction
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from datasets import Dataset, IterableDataset
from transformers.utils import is_peft_available

from trl.trainer.grpo_config import GRPOConfig
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

import numpy as np

if is_peft_available():
    from peft import PEFTConfig, get_peft_model

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[List, List], List[float]]]

class Qwen2VLGRPOTrainer(Trainer):
    """
    GRPO Trainer adapted for Qwen2-VL model with vision-language support.
    
    This trainer extends the standard GRPO algorithm to support multimodal inputs
    (text + images) using the Qwen2-VL architecture.
    
    Args:
        model (`Union[str, PreTrainedModel]`):
            Qwen2-VL model to be trained.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions for computing rewards. Can be a single function or list of functions, or reward model name.
        args ([`GRPOConfig`], *optional*):
            Training configuration.
        attn_implementation (`str`, *optional*):
            Attention implementation type ('flash_attention_2', 'sdpa', etc.).
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration for parameter-efficient training.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*):
            Processor/tokenizer for Qwen2-VL.
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*):
            Processing classes for reward models.
        max_pixels (`int`, *optional*, defaults to 12845056):
            Maximum number of pixels for image processing.
        min_pixels (`int`, *optional*, defaults to 3136):
            Minimum number of pixels for image processing.
        train_dataset ([`~datasets.Dataset`]):
            Training dataset with column: ['prompt', 'gt_agents', 'gt_dataset'].
        eval_dataset ([`~datasets.Dataset`], *optional*):
            Evaluation dataset.
        callbacks (`list[TrainerCallback]`, *optional*):
            Additional callbacks.
        optimizers (`tuple`, *optional*):
            Optimizer and scheduler tuple.
    """

    def __init__(
            self, 
            model: Union[str, PreTrainedModel],
            reward_funcs: Union[RewardFunc, List[RewardFunc]],
            args: GRPOConfig=None,
            attn_implementation: str="flash_attention_2",
            peft_config: Optional["PEFTConfig"]=None,
            processing_class: Optional[PreTrainedTokenizerBase]=None,
            reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, List[PreTrainedTokenizerBase]]]=None,
            max_pixels: int=12845056,
            min_pixels: int=3136,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        ):
        if args is None:
            self.model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]  # 取最后的一段名字
            args = GRPOConfig(f"{model_name}-grpo")  # 这里只是定义了一个运行的名字，实际的配置参数在这里都是默认值

        # 开始配置training model的部分

        # 首先尝试从args里获取一些模型基础参数，来源是prl库允许的GRPOConfig里增加的model_init_kwargs字段，这里会存储一些模型初始化的参数
        model_init_kwargs = args.model_init_kwargs if hasattr(args, "model_init_kwargs") else {}
        # 单独获取attn_implementation参数，因为它与硬件/模型训练需求等相关，需要特别指定
        model_init_kwargs["attn_implementation"] = attn_implementation
        # 开始正式加载模型
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            # 因为 用户传入的 dtype 可能是 string / torch.dtype / None，多种格式，而 HF 的 from_pretrained() 只接受合法的 torch.dtype 或 "auto"
            if isinstance(torch_dtype, str):
                if torch_dtype == "auto":
                    pass
                else:
                    torch_dtype = getattr(torch, torch_dtype)
                    model_init_kwargs["torch_dtype"] = torch_dtype
            elif isinstance(torch_dtype, torch.dtype) or torch_dtype is None:
                pass
            else:
                raise ValueError("Line94: torch_dtype must be str, torch.dtype, or None")
            # 根据gradient_checkpointing参数调整use_cache参数
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            # 加载模型
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError("Line118: When passing a model instance, model_init_kwargs should be None.")
        
        # peft配置加载
        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # 加载reference model
        # 目的是加载一个参数冻结的模型，用于计算KL散度
        # 首先检查DeepSpeed ZeRO-3是否启用，因为DeepSpeed ZeRO-3 下，模型的权重被“分布式切片存储”，不能直接copy
        if is_deepspeed_zero3_enabled():
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # 如果没有使用PEFT，则可以直接复制模型
            self.ref_model = create_reference_model(model)
        else:
            # 如果使用了PEFT，可以直接通过解除adapter的方式得到冻结的原始模型，不需要额外reference model
            self.ref_model = None

        # 加载reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            # 这里的目的是：当reward_func是字符串时，加载对应的预训练模型，否则直接使用传入的模型或函数
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # 下面加载preprocessing 模型，分为训练模型的processor 和 reward model的processor
        # 先看training model的processor
        if processing_class is None:
            if "Qwen2-VL" in model_id:
                self.processor = AutoProcessor.from_pretrained(model_id)
                processing_class.pad_token_id = self.processor.tokenizer.pad_token_id
                processing_class.eos_token_id = self.processor.tokenizer.eos_token_id
                processing_class.image_processor.max_pixels = max_pixels
                processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id
        
        # 再加载 reward model的processor，这段逻辑完全是为了处理 reward_func 是 HF 模型的情况
        if reward_processing_classes is None:
            reward_procesing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_procesing_classes, list):
            reward_procesing_classes = [reward_procesing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("Line170: reward_processing_classes must be the same length as reward_funcs.")
        for i, (reward_func, reward_processing_class) in enumerate(zip(reward_funcs, reward_processing_classes)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features
        
        # 这里需要处理一下trainer可能出现的数据集缺失inputs_id的警告，我们的train_dataset里没有inputs_id字段,和示例代码保持一致
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        # 处理 Trainer 的初始化参数
        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper # 最大生成长度
        self.num_generations = args.num_generations  # = G in the GRPO paper  # 每个输入生成多少个样本
        self.temperature = args.temperature
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=self.temperature,  # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        """
        Trainer 会检查：模型是否愿意接收 “loss 相关的 kwarg”（例如 labels=...、return_loss=True）
            如果模型接受 loss 的 kwarg，则 Trainer 会做一些特殊处理；
            如果模型不接受，则 Trainer 用另一套路径来处理 loss。    
        关键变量是：
            self.model_accepts_loss_kwargs
        Trainer 用它决定：
            是否给 model 传递额外的 loss 参数
            是否自动缩放 loss（loss scaling）
            是否按 gradient accumulation 正确平均 loss
        """
        self.model_accepts_loss_kwargs = False

        # 核心目的就是把 reference model 准备好，确保在训练时可以正确在加速器上运行
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # 关于signature的设置，这里的signature_columns是为了在Trainer中标记输入数据的特征列
        # Trainer会在模型输入时，自动将这些列转换为tensor，并传递给模型
        # 若输入数据集里有别的列，会在输入模型前删去，只保留在signature_columns里指定的列
        # 不用担心gt列会被删去，因为这里只是为输入model的数据做标记，gt列不会从数据集上消失，后续reward计算时会单独处理
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    # 为了跳过默认的 tensor/device 处理
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        我们这里重写 compute_loss 方法，以实现 GRPO 的损失计算逻辑。
        Compute the GRPO loss for a batch of inputs.
        Args:
            model (`PreTrainedModel`):
                The model to compute the loss for.
            inputs (`dict`):
                A batch of inputs from the dataset.
            return_outputs (`bool`, *optional*, defaults to `False`):
                Whether to return the model outputs along with the loss.
            num_items_in_batch (`int`, *optional*, defaults to `None`):
                Number of items in the batch (used for logging).
        """

        # 我们这里不需要返回outputs
        assert return_outputs is False, "Line333: return_outputs must be False in GRPO"

        # 处理模型输入
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)['prompt'] for example in inputs]  # 注：这里我观察到self.processing_class不存在,以前并未放入self中
        images = [x["image"] for x in inputs] if "image" in inputs[0] else None
        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            # 去掉 accelerator 封装，但保留 PeftModel 和 LoRA 权重
            # prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)

            # Generate N times, each generate one with the temp_generation_config , stack the output_ids to prompt_completion_ids, pad the empty places with number 151613
            num_generations = self.generation_config.num_return_sequences
            temp_generation_config = copy.deepcopy(self.generation_config)
            temp_generation_config.num_return_sequences = 1

            all_completions = []

            for i in range(num_generations):  # -1 because we already have one generation
                completion = unwrapped_model.generate(**prompt_inputs, generation_config=temp_generation_config)
                all_completions.append(completion)

            # Stack all completions and pad if needed
            max_length = max(completion.size(1) for completion in all_completions)
            padded_completions = []

            for completion in all_completions:
                if completion.size(1) < max_length:
                    padding = torch.full(
                        (completion.size(0), max_length - completion.size(1)),
                        self.processing_class.tokenizer.pad_token_id,
                        dtype=completion.dtype,
                        device=completion.device,
                    )
                    padded_completion = torch.cat([completion, padding], dim=1)
                else:
                    padded_completion = completion
                padded_completions.append(padded_completion)

            # Stack all padded completions，变成了 (batch_size * num_generations, max_length)
            prompt_completion_ids = torch.cat(padded_completions, dim=0)

        # 因为generate的结果是input + output，我们只想要得到模型输出部分，所以需要切分
        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # 完成一个函数，用于计算生成的sequence的对数概率
        def get_per_token_logps(model, input_ids, **kwargs):
            logits = model(input_ids=input_ids, **kwargs).logits  #(batch_size, seq_len, vocab_size)
            # 由于上述logits的每一个预测的都是后一个token在词表中的概率，所以需要把最后一个logit去掉，它不具备预测意义，因为它后面没有词
            logits = logits[:, :-1, :]  #(batch_size, seq_len-1, vocab_size)
            # 同时input_ids也需要去掉第一个token，因为第一个token没有前驱，无法计算概率
            input_ids = input_ids[:, 1:]  #(batch_size, seq_len-1

            # 开始计算 log probabilities
            total_log_probs = []
            for cur_logits, cur_input_ids in zip(logits, input_ids):
                # cur_logits: (seq_len-1, vocab_size), 是整个batch里当前正在处理的一个sequence的每个token的在词表里各个id的logits
                # cur_input_ids: (seq_len-1), 是当前sequence的每个token的真实id
                log_probs = nn.functional.log_softmax(cur_logits, dim=-1)  #(seq_len-1, vocab_size)
                # 选择对应token的log prob
                # # 1，先改变input_ids的shape，变成(seq_len-1, 1)
                cur_input_ids = cur_input_ids.unsqueeze(-1)  #(seq_len-1, 1)
                # # 2，使用gather函数从log_probs里选择对应token的log prob
                selected_log_probs = log_probs.gather(dim=-1, index=cur_input_ids)  #(seq_len-1, 1)
                total_log_probs.append(selected_log_probs.squeeze(-1))
            return total_log_probs  #(batch_size, seq_len-1) 其中是每一个batch里每个sequence的每个token的log prob
        
        # 这里我们已经使用prompt_inputs生成了多个completions，接下来我们需要计算这些completions的log概率
        # 为了计算log概率，我们需要把prompt_inputs里与文本相关的部分去掉，只保留图像相关的部分
        prompt_inputs.pop("input_ids")
        prompt_inputs.pop("attention_mask")
        # 由于之前我们一个prompt生成了多个completion，现在需要把prompt_inputs里的每个图像相关信息复制num_generations次，以匹配prompt_completion_ids的shape
        prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].repeat(len(prompt_completion_ids), 1)
        prompt_inputs["image_grid_thw"] = prompt_inputs["image_grid_thw"].repeat(len(prompt_completion_ids), 1)
        # 计算log概率，这里！！！注意，我发现删去的attention_mask并没有被新的东西替代，所以要么模型自己能处理，要么这里就是个bug
        per_token_logps = get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)，删去log概率里与prompt相关的部分
        per_token_logps = per_token_logps[:, prompt_length - 1 :]

        # 同理，计算reference model的log概率
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = get_per_token_logps(self.ref_model, prompt_completion_ids)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = get_per_token_logps(model, prompt_completion_ids)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1  #(batch_size * num_generations, seq_len)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                # 目的: 获取除了 prompt 和 completion 之外的所有字段(如 image, metadata 等)
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                # 重复这些字段的值，以匹配生成的数量
                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                # 这里需要修改，与我的reward function调用方式保持一致
                output_reward_func = reward_func(completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Sum the rewards from all reward functions，直接相加
        rewards = rewards_per_func.sum(dim=1)
        # GRPO 的核心思想是：每个 prompt 生成多个样本（num_generations），使用 同组内部的均值和标准差 来计算 advantage。
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        # 把组均值和方差复制回每一个 completion，重复 num_generations 次
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        # GRPO 的核心：当前样本奖励相对于同组平均值差多少？归一化后就是 advantage
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        # 计算 per-token loss
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        # 加上 KL 惩罚（per_token_kl）
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        # Mask + 求平均得到最终 loss
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        # Logging 记录指标
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss


    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()


    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
             
