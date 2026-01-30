from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import re
import os
import logging
from datetime import datetime
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, TrlParser, ModelConfig
import torch

logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """
    Base script arguments.
    """
    dataset_name: str = field(
        metadata={"help": "Name of the dataset to use"}
    )
    dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "Configuration name of the dataset"}
    )
    dataset_train_split: str = field(
        default="train",
        metadata={"help": "Train split name"}
    )
    dataset_test_split: str = field(
        default="test",
        metadata={"help": "Test split name"}
    )


@dataclass
class DataFilteringGRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the data filtering agent selection GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'agent_selection', 'filtering_quality', 'format', 'combined'.
        agent_selection_weight (`float`):
            Weight for agent selection accuracy reward (default: 0.5).
        filtering_quality_weight (`float`):
            Weight for filtering quality reward (default: 0.5).
        available_agents (`list[str]`):
            List of available agent names that can be selected.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["combined", "format"],
        metadata={"help": "List of reward functions. Possible values: 'agent_selection', 'filtering_quality', 'format', 'combined'"},
    )
    agent_selection_weight: float = field(
        default=0.5,
        metadata={"help": "Weight for agent selection accuracy reward"},
    )
    filtering_quality_weight: float = field(
        default=0.5,
        metadata={"help": "Weight for filtering quality reward"},
    )
    available_agents: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "List of available agent names"},
    )


def parse_agent_selection(content: str) -> Dict[str, Any]:
    """
    Parse the model output to extract agent names and prompts.
    
    Returns:
        Dict with keys: 'agents' (list of agent names), 'prompts' (dict mapping agent_name to prompt)
    """
    result = {
        'agents': [],
        'prompts': {},
        'reasoning': ''
    }
    
    try:
        # Extract reasoning
        reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", content, re.DOTALL)
        if reasoning_match:
            result['reasoning'] = reasoning_match.group(1).strip()
        
        # Extract agents section
        agents_match = re.search(r"<agents>(.*?)</agents>", content, re.DOTALL)
        if not agents_match:
            return result
        
        agents_content = agents_match.group(1)
        
        # Extract individual agents
        agent_pattern = r"<agent>(.*?)</agent>"
        agent_matches = re.finditer(agent_pattern, agents_content, re.DOTALL)
        
        for agent_match in agent_matches:
            agent_content = agent_match.group(1)
            
            # Extract agent name
            name_match = re.search(r"<name>(.*?)</name>", agent_content, re.DOTALL)
            if not name_match:
                continue
            agent_name = name_match.group(1).strip()
            
            # Extract prompt
            prompt_match = re.search(r"<prompt>(.*?)</prompt>", agent_content, re.DOTALL)
            if prompt_match:
                prompt = prompt_match.group(1).strip()
                result['agents'].append(agent_name)
                result['prompts'][agent_name] = prompt
            else:
                result['agents'].append(agent_name)
                result['prompts'][agent_name] = ""
                
    except Exception as e:
        pass  # Return empty result if parsing fails
    
    return result


def agent_executor(agent_names, prompts, original_dataset, **kwargs):
    """
    Placeholder for agent execution function.
    Execute the selected agents with their prompts on the dataset.
    
    Args:
        agent_names: List of agent names to execute
        prompts: Dict mapping agent_name to prompt
        original_dataset: The dataset to filter
        
    Returns: 
        list of filtered data IDs/indices
    """
    # TODO: Implement actual agent execution logic
    # This is a placeholder that should be replaced with actual implementation
    return []


def agent_selection_reward(completions, ground_truth_agents, **kwargs):
    """
    Reward function that evaluates agent selection accuracy.
    
    Args:
        completions: Model completions
        ground_truth_agents: List of ground truth agent names for each example
        
    Returns:
        List of rewards (0.0 to 1.0)
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    for content, gt_agents in zip(contents, ground_truth_agents):
        reward = 0.0
        
        try:
            parsed = parse_agent_selection(content)
            selected_agents = set(parsed['agents'])
            gt_agents_set = set(gt_agents) if isinstance(gt_agents, list) else set([gt_agents])
            
            # Calculate F1 score for agent selection
            if len(selected_agents) == 0 and len(gt_agents_set) == 0:
                reward = 1.0
            elif len(selected_agents) == 0 or len(gt_agents_set) == 0:
                reward = 0.0
            else:
                # True positives: agents in both selected and ground truth
                tp = len(selected_agents & gt_agents_set)
                # False positives: agents selected but not in ground truth
                fp = len(selected_agents - gt_agents_set)
                # False negatives: agents in ground truth but not selected
                fn = len(gt_agents_set - selected_agents)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                if precision + recall > 0:
                    f1_score = 2 * (precision * recall) / (precision + recall)
                    reward = f1_score
                else:
                    reward = 0.0
                    
        except Exception as e:
            reward = 0.0
        
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "debug.log")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Agent Selection Reward: {reward:.4f} -------------\n")
                f.write(f"Selected agents: {parsed.get('agents', [])}\n")
                f.write(f"Ground truth agents: {gt_agents}\n")
                f.write(f"Content: {content[:500]}...\n\n")
    
    return rewards


def filtering_quality_reward(completions, ground_truth_filtered_data, original_dataset=None, **kwargs):
    """
    Reward function that evaluates the quality of filtered data.
    
    Args:
        completions: Model completions
        ground_truth_filtered_data: Ground truth filtered dataset indices or data IDs
        original_dataset: The original dataset to filter (optional)
        
    Returns:
        List of rewards (0.0 to 1.0)
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    for content, gt_filtered in zip(contents, ground_truth_filtered_data):
        reward = 0.0
        
        try:
            parsed = parse_agent_selection(content)
            
            # Execute the agents to get filtered data
            filtered_data = agent_executor(parsed['agents'], parsed['prompts'], 
                                          original_dataset=original_dataset, **kwargs)
            
            # Convert to sets for comparison (assuming data is represented by IDs or indices)
            filtered_set = set(filtered_data) if filtered_data else set()
            gt_filtered_set = set(gt_filtered) if isinstance(gt_filtered, list) else set([gt_filtered])
            
            # Calculate F1 score for filtering quality
            if len(filtered_set) == 0 and len(gt_filtered_set) == 0:
                reward = 1.0
            elif len(filtered_set) == 0 or len(gt_filtered_set) == 0:
                reward = 0.0
            else:
                tp = len(filtered_set & gt_filtered_set)
                fp = len(filtered_set - gt_filtered_set)
                fn = len(gt_filtered_set - filtered_set)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                if precision + recall > 0:
                    f1_score = 2 * (precision * recall) / (precision + recall)
                    reward = f1_score
                else:
                    reward = 0.0
                    
        except Exception as e:
            reward = 0.0
        
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "debug.log")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Filtering Quality Reward: {reward:.4f} -------------\n")
                f.write(f"Filtered data size: {len(filtered_set)}\n")
                f.write(f"Ground truth size: {len(gt_filtered_set)}\n")
                f.write(f"Overlap: {len(filtered_set & gt_filtered_set)}\n\n")
    
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has the correct XML format."""
    pattern = r"<selection>.*?<reasoning>.*?</reasoning>.*?<agents>.*?</agents>.*?</selection>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def combined_reward(completions, ground_truth_agents, ground_truth_filtered_data, 
                   agent_weight=0.5, filtering_weight=0.5, **kwargs):
    """
    Combined reward function that weighs both agent selection and filtering quality.
    
    Args:
        agent_weight: Weight for agent selection reward (default: 0.5)
        filtering_weight: Weight for filtering quality reward (default: 0.5)
    """
    agent_rewards = agent_selection_reward(completions, ground_truth_agents, **kwargs)
    filtering_rewards = filtering_quality_reward(completions, ground_truth_filtered_data, **kwargs)
    
    combined = [
        agent_weight * ar + filtering_weight * fr 
        for ar, fr in zip(agent_rewards, filtering_rewards)
    ]
    
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH", "debug.log")
        with open(log_path, "a") as f:
            f.write(f"------------- {current_time} Combined Reward -------------\n")
            f.write(f"Agent rewards: {agent_rewards}\n")
            f.write(f"Filtering rewards: {filtering_rewards}\n")
            f.write(f"Combined rewards: {combined}\n\n")
    
    return combined


reward_funcs_registry = {
    "agent_selection": agent_selection_reward,
    "filtering_quality": filtering_quality_reward,
    "format": format_reward,
    "combined": combined_reward,
}


def create_system_prompt(available_agents):
    """Create system prompt with available agents listed."""
    agents_list = "\n".join([f"- {agent}" for agent in available_agents])
    
    return (
        "You are an expert data filtering system. Given a dataset report, you must:\n"
        "1. Analyze the data characteristics and quality issues\n"
        "2. Select the most appropriate filtering agents from the available options\n"
        "3. Generate specific prompts for each selected agent\n\n"
        f"Available agents:\n{agents_list}\n\n"
        "Output your response in the following format:\n"
        "<selection>\n"
        "<reasoning>\n"
        "[Your analysis and reasoning for agent selection]\n"
        "</reasoning>\n"
        "<agents>\n"
        "<agent>\n"
        "<name>agent_name</name>\n"
        "<prompt>specific filtering instructions</prompt>\n"
        "</agent>\n"
        "... more agents as needed ...\n"
        "</agents>\n"
        "</selection>"
    )


def main(script_args, training_args, model_args):
    """
    Main training function for data filtering agent selection.
    """
    # Get reward functions with weights
    reward_funcs = []
    for func_name in script_args.reward_funcs:
        if func_name == "combined":
            # Create a lambda to pass weights to combined_reward
            reward_func = lambda completions, **kwargs: combined_reward(
                completions,
                agent_weight=script_args.agent_selection_weight,
                filtering_weight=script_args.filtering_quality_weight,
                **kwargs
            )
            reward_funcs.append(reward_func)
        else:
            reward_funcs.append(reward_funcs_registry[func_name])

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Create system prompt with available agents
    SYSTEM_PROMPT = create_system_prompt(script_args.available_agents)

    # Format dataset into conversation format
    def make_conversation(example):
        """
        Expected dataset format:
        {
            'report': str,  # Dataset report text
            'ground_truth_agents': list[str],  # List of correct agent names
            'ground_truth_filtered_data': list[int/str],  # List of data IDs/indices that should be kept
        }
        """
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"REPORT IS :\n {example['report']}"},
            ],
            # Pass ground truth for reward calculation
            "ground_truth_agents": example.get("ground_truth_agents", []),
            "ground_truth_filtered_data": example.get("ground_truth_filtered_data", []),
        }

    # Map dataset to conversation format
    dataset = dataset.map(make_conversation, remove_columns=dataset[script_args.dataset_train_split].column_names)

    # Initialize the GRPO trainer
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args) if hasattr(model_args, 'use_peft') and model_args.use_peft else None,
    )

    # Train the model
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
    
    # 自动注册版本（如果可用）
    try:
        import sys
        from pathlib import Path
        # 添加父目录到路径，以便导入 version 模块
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from version.version_manager import VersionManager
        
        # 获取最终奖励（如果可用）
        final_reward = None
        if hasattr(trainer, "_metrics") and "reward" in trainer._metrics:
            rewards = trainer._metrics["reward"]
            if rewards:
                final_reward = sum(rewards) / len(rewards)
        
        # 注册版本
        manager = VersionManager()
        version = manager.register_version(
            model_path=str(training_args.output_dir),
            config_path=getattr(script_args, "config_path", None),
            training_epochs=training_args.num_train_epochs,
            final_reward=final_reward,
            description=f"GRPO训练 - {script_args.dataset_name}",
            metadata={
                "reward_funcs": script_args.reward_funcs,
                "available_agents": script_args.available_agents,
            }
        )
        print(f"\n✓ 模型已注册为版本: {version}")
    except ImportError:
        logger.warning("version_manager 未找到，跳过版本注册")
    except Exception as e:
        logger.warning(f"版本注册失败: {e}")


def get_peft_config(model_args):
    """
    Get PEFT configuration if needed.
    You may need to implement this based on your model configuration.
    """
    from peft import LoraConfig
    
    if hasattr(model_args, 'use_peft') and model_args.use_peft:
        return LoraConfig(
            r=model_args.lora_r if hasattr(model_args, 'lora_r') else 16,
            lora_alpha=model_args.lora_alpha if hasattr(model_args, 'lora_alpha') else 32,
            lora_dropout=model_args.lora_dropout if hasattr(model_args, 'lora_dropout') else 0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    return None


if __name__ == "__main__":
    parser = TrlParser((DataFilteringGRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    # Validate that available_agents is provided
    if not script_args.available_agents:
        raise ValueError("Please provide --available_agents as a list of agent names")
    
    main(script_args, training_args, model_args)


"""
python train_grpo.py --config config.yaml

{
  "report": "Dataset contains 10000 samples with high duplication rate (35%). Quality issues: 15% samples have incomplete information. Average text length: 250 tokens.",
  "ground_truth_agents": ["deduplication_filter", "quality_filter"],
  "ground_truth_filtered_data": [0, 1, 5, 7, 9, 12, ...]
}
"""