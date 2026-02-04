"""
VQA问题生成系统主模块（预填充对象版本）
实现完整的6步流程，但使用预填充对象而不是对象选择

注意：claim方式不再提取对象，直接将claim作为问题生成prompt的一部分
"""
import json
import re
import random
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from contextlib import AsyncExitStack

# 复用原有模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from QA_Generator.question.core.config_loader import ConfigLoader
from QA_Generator.question.core.slot_filler import SlotFiller
from QA_Generator.question.core.validator import QuestionValidator
from QA_Generator.clients.gemini_client import GeminiClient
from QA_Generator.clients.async_client import AsyncGeminiClient

# 导入预填充专用模块
from .object_prefill import PrefillProcessor
from .prefill_processor_simplified import PrefillProcessorSimplified
from .question_generator_prefill import QuestionGeneratorPrefill


class VQAGeneratorPrefill:
    """
    VQA问题生成器主类（预填充对象版本）
    
    与原始版本的主要区别：
    1. STEP 2改为处理预填充对象（而不是对象选择）
    2. 问题生成时必须使用预填充对象
    3. 输入数据需要包含prefill字段（claim或target_object）
    """
    
    def __init__(
        self,
        config_path: Path,
        gemini_client: Optional[GeminiClient] = None,
        failed_selection_dir: Optional[Path] = None,
        enable_validation_exemptions: bool = False,
    ):
        """
        初始化VQA生成器（预填充版本）
        
        Args:
            config_path: 配置文件路径
            gemini_client: Gemini客户端实例（可选）
            failed_selection_dir: 失败案例存储目录（可选，预填充版本可能不需要）
        """
        self.config_loader = ConfigLoader(config_path)
        self.gemini_client = gemini_client or GeminiClient()
        
        # 初始化各个模块
        self.prefill_processor = PrefillProcessor(self.gemini_client)  # 保留用于向后兼容
        self.prefill_processor_simplified = PrefillProcessorSimplified()  # 新的简化版本
        self.slot_filler = SlotFiller(self.gemini_client)
        # question_generator 在 global_constraints 加载后初始化（见下方）
        self.validator = QuestionValidator(
            self.gemini_client,
            enable_validation_exemptions=enable_validation_exemptions,
        )
        
        # 获取策略配置
        self.global_constraints = self.config_loader.get_global_constraints()
        self.generation_policy = self.config_loader.get_generation_policy()
        self.question_type_ratio = self.config_loader.get_question_type_ratio()
        
        # 问题生成器需要全局约束（用于将validation_rules加入生成prompt）
        self.question_generator = QuestionGeneratorPrefill(
            self.gemini_client,
            global_constraints=self.global_constraints,
        )
        
        # 失败案例存储目录（预填充版本可能不需要，但保留接口）
        self.failed_selection_dir = failed_selection_dir
        if self.failed_selection_dir:
            self.failed_selection_dir.mkdir(parents=True, exist_ok=True)
    
    def process_image_pipeline_pair(
        self,
        image_input: Any,
        pipeline_name: str,
        prefill_input: Dict[str, Any],  # 预填充输入（必需）
        metadata: Optional[Dict[str, Any]] = None,
        debug_record: Optional[Dict[str, Any]] = None,
    ) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        处理单个图片-pipeline对，生成VQA问题（使用预填充对象）
        
        严格遵循6步流程：
        1. 加载Pipeline规范
        2. 处理预填充对象（替代对象选择）
        3. 槽位填充
        4. 问题生成（使用预填充对象）
        5. 验证（可选，当前已跳过）
        6. 输出
        
        Args:
            image_input: 图片输入（路径、base64、bytes等）
            pipeline_name: Pipeline名称
            prefill_input: 预填充输入，包含以下字段之一：
                - "claim": 一句包含基于对象的该图片的claim
                - "target_object": 目标对象名字
                - "target_object_info": 目标对象详细信息（可选）
            metadata: 可选的元数据
            
        Returns:
            (成功结果, 错误/丢弃信息)
            如果成功: (结果字典, None)
            如果失败: (None, 错误信息字典)
        """
        error_info = {
            "pipeline_name": pipeline_name,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "error_stage": None,
            "error_reason": None
        }
        
        try:
            # STEP 1: 加载Pipeline规范
            pipeline_config = self.config_loader.get_pipeline_config(pipeline_name)
            if not pipeline_config:
                error_info["error_stage"] = "config_loading"
                error_info["error_reason"] = f"Pipeline '{pipeline_name}' 不存在"
                print(f"[WARNING] {error_info['error_reason']}，跳过")
                if debug_record is not None:
                    debug_record["error_stage"] = error_info["error_stage"]
                    debug_record["error_reason"] = error_info["error_reason"]
                return None, error_info
            
            # STEP 2: 处理预填充输入（简化版本）
            prefill_info = None
            try:
                # 检查是否使用新的简化格式（包含 prefilled_values）
                if "prefilled_values" in prefill_input:
                    prefill_info = self.prefill_processor_simplified.process_prefill(
                        prefill_input=prefill_input,
                        image_input=image_input,
                        pipeline_config=pipeline_config
                    )
                else:
                    # 向后兼容：处理旧格式（仅用于兼容可能存在的旧数据文件）
                    # 注意：新流程统一使用 claim + prefilled_values，不再使用 target_object
                    prefill_object = self.prefill_processor.process_prefill(
                        prefill_input=prefill_input,
                        image_input=image_input,
                        pipeline_config=pipeline_config
                    )
                    if prefill_object:
                        # 转换为新格式：PrefillProcessor 已经处理了优先级，直接使用 name
                        prefill_info = {
                            "claim": prefill_object.get("claim", ""),
                            "prefilled_values": {}
                        }
                        # PrefillProcessor 返回的 name 字段已经包含了正确的值（优先来自 target_object）
                        if prefill_object.get("name"):
                            prefill_info["prefilled_values"]["OBJECT_A"] = prefill_object["name"]
                        if prefill_object.get("other_object"):
                            prefill_info["prefilled_values"]["OBJECT_B"] = prefill_object["other_object"]
                
                if not prefill_info or not prefill_info.get("claim"):
                    error_info["error_stage"] = "prefill_processing"
                    error_info["error_reason"] = "预填充处理失败或缺少 claim"
                    print(f"[ERROR] {error_info['error_reason']}")
                    if debug_record is not None:
                        debug_record["error_stage"] = error_info["error_stage"]
                        debug_record["error_reason"] = error_info["error_reason"]
                    return None, error_info
                    
            except Exception as e:
                error_info["error_stage"] = "prefill_processing"
                error_info["error_reason"] = f"预填充处理过程出错: {str(e)}"
                print(f"[ERROR] {error_info['error_reason']}")
                if debug_record is not None:
                    debug_record["error_stage"] = error_info["error_stage"]
                    debug_record["error_reason"] = error_info["error_reason"]
                return None, error_info
            
            # STEP 3: 槽位填充
            # 注意：同步版本暂不支持简化流程（需要异步），这里使用旧的流程
            # 如果使用简化格式，需要转换为旧格式供 fill_slots 使用
            try:
                # 构建 selected_object 用于向后兼容（如果需要）
                selected_object = None
                if prefill_info and prefill_info.get("prefilled_values"):
                    # 尝试从 prefilled_values 构建 selected_object
                    prefilled = prefill_info["prefilled_values"]
                    if "OBJECT_A" in prefilled:
                        selected_object = {"name": prefilled["OBJECT_A"]}
                
                slots = self.slot_filler.fill_slots(
                    image_input=image_input,
                    pipeline_config=pipeline_config,
                    selected_object=selected_object
                )
                
                if slots is None:
                    error_info["error_stage"] = "slot_filling"
                    error_info["error_reason"] = "槽位填充失败（必需槽位无法解析）"
                    if debug_record is not None:
                        debug_record["error_stage"] = error_info["error_stage"]
                        debug_record["error_reason"] = error_info["error_reason"]
                    return None, error_info
            except Exception as e:
                error_info["error_stage"] = "slot_filling"
                error_info["error_reason"] = f"槽位填充过程出错: {str(e)}"
                print(f"[ERROR] {error_info['error_reason']}")
                if debug_record is not None:
                    debug_record["error_stage"] = error_info["error_stage"]
                    debug_record["error_reason"] = error_info["error_reason"]
                return None, error_info
            
            # STEP 4: 问题生成（使用预填充对象）
            question_type = self._select_question_type()
            
            # 构建 prefill_object 用于问题生成
            prefill_object = {
                "claim": prefill_info.get("claim", ""),
                "prefilled_values": prefill_info.get("prefilled_values", {}),
                "source": "simplified" if "prefilled_values" in prefill_input else "legacy"
            }
            
            try:
                question = self.question_generator.generate_question(
                    image_input=image_input,
                    pipeline_config=pipeline_config,
                    slots=slots,
                    prefill_object=prefill_object,
                    question_type=question_type
                )
                
                if not question:
                    error_info["error_stage"] = "question_generation"
                    error_info["error_reason"] = "问题生成失败（返回空）"
                    if debug_record is not None:
                        debug_record["error_stage"] = error_info["error_stage"]
                        debug_record["error_reason"] = error_info["error_reason"]
                    return None, error_info
            except Exception as e:
                error_info["error_stage"] = "question_generation"
                error_info["error_reason"] = f"问题生成过程出错: {str(e)}"
                print(f"[ERROR] {error_info['error_reason']}")
                if debug_record is not None:
                    debug_record["error_stage"] = error_info["error_stage"]
                    debug_record["error_reason"] = error_info["error_reason"]
                return None, error_info
            
            # STEP 5: 输出（跳过验证以提高效率）
            result = {
                "pipeline_name": pipeline_name,
                "pipeline_intent": pipeline_config.get("intent", ""),
                "question": question,
                "question_type": question_type,
                "answer_type": pipeline_config.get("answer_type", ""),
                "slots": slots,
                "prefill_object": prefill_object,  # 保存预填充对象信息
                "prefill_source": prefill_object.get("source"),  # 保存来源
                "validation_reason": "skipped",
                "timestamp": datetime.now().isoformat()
            }
            if debug_record is not None:
                debug_record["question_type"] = question_type
                debug_record["prefill_object"] = prefill_object
                debug_record["slots"] = slots
                debug_record["question"] = question
                debug_record["error_stage"] = None
                debug_record["error_reason"] = None
            
            return result, None
            
        except Exception as e:
            error_info["error_stage"] = "unknown"
            error_info["error_reason"] = f"未知错误: {str(e)}"
            print(f"[ERROR] {error_info['error_reason']}")
            if debug_record is not None:
                debug_record["error_stage"] = error_info["error_stage"]
                debug_record["error_reason"] = error_info["error_reason"]
            return None, error_info
    
    def process_data_file(
        self,
        input_file: Path,
        output_file: Path,
        pipeline_names: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        failed_selection_dir: Optional[Path] = None,
        debug_output_dir: Optional[Path] = None,
        suppress_progress: bool = False,
    ) -> None:
        """
        处理数据文件，为每张图片生成VQA问题（使用预填充对象）
        
        Args:
            input_file: 输入JSON文件路径（必须包含prefill字段）
            output_file: 输出JSON文件路径
            pipeline_names: 要使用的pipeline列表（None表示使用所有）
            max_samples: 最大处理样本数（None表示全部）
            failed_selection_dir: 失败案例存储目录（可选）
        """
        if failed_selection_dir is not None:
            self.failed_selection_dir = failed_selection_dir
            self.failed_selection_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 读取输入文件: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError(f"输入文件应该包含一个数组，但得到: {type(data)}")
        
        # 确定要使用的pipeline
        if pipeline_names is None:
            pipeline_names = self.config_loader.list_pipelines()
        
        print(f"[INFO] 使用pipelines: {pipeline_names}")
        print(f"[INFO] 总记录数: {len(data)}")
        
        if max_samples:
            data = data[:max_samples]
            print(f"[INFO] 限制处理前 {max_samples} 条记录")
        
        # 处理每条记录
        results = []
        errors = []
        debug_records: List[Dict[str, Any]] = []
        total_processed = 0
        total_discarded = 0
        
        for idx, record in enumerate(data, 1):
            source_a = record.get("source_a", {})
            if not source_a:
                error_info = {
                    "record_index": idx,
                    "id": record.get("id"),
                    "error_stage": "data_loading",
                    "error_reason": "记录没有source_a",
                    "timestamp": datetime.now().isoformat()
                }
                errors.append(error_info)
                print(f"[WARNING] 记录 {idx} 没有source_a，跳过")
                continue
            
            # 提取图片输入
            image_input = self._extract_image_input(source_a)
            if image_input is None:
                error_info = {
                    "record_index": idx,
                    "id": record.get("id"),
                    "source_a_id": source_a.get("id"),
                    "error_stage": "data_loading",
                    "error_reason": "无法提取图片输入",
                    "timestamp": datetime.now().isoformat()
                }
                errors.append(error_info)
                print(f"[WARNING] 记录 {idx} 无法提取图片，跳过")
                continue
            
            # 提取预填充输入（必需）
            prefill_input = record.get("prefill")
            if not prefill_input:
                error_info = {
                    "record_index": idx,
                    "id": record.get("id"),
                    "error_stage": "data_loading",
                    "error_reason": "记录缺少prefill字段（需要包含'claim'或'target_object'）",
                    "timestamp": datetime.now().isoformat()
                }
                errors.append(error_info)
                print(f"[WARNING] 记录 {idx} 缺少prefill字段，跳过")
                continue
            
            # 确定该记录应该使用的pipeline
            record_pipeline = self._extract_pipeline_from_record(record)
            pipelines_to_use = [record_pipeline] if record_pipeline else pipeline_names
            
            # 为确定的pipeline生成问题
            for pipeline_name in pipelines_to_use:
                total_processed += 1
                debug_record = {
                    "record_index": idx,
                    "sample_index": record.get("sample_index"),
                    "id": record.get("id"),
                    "pipeline_name": pipeline_name,
                    "prefill_input": prefill_input,
                    "input_record": record,
                    "timestamp": datetime.now().isoformat(),
                }
                
                result, error_info = self.process_image_pipeline_pair(
                    image_input=image_input,
                    pipeline_name=pipeline_name,
                    prefill_input=prefill_input,  # 传入预填充输入
                    metadata={"record_index": idx, "id": record.get("id")},
                    debug_record=debug_record,
                )
                
                if result:
                    # 添加原始数据信息
                    result["sample_index"] = record.get("sample_index")
                    result["id"] = record.get("id")
                    result["source_a_id"] = source_a.get("id")
                    # 添加图片的base64编码
                    image_base64 = self._extract_image_base64(source_a, image_input)
                    if image_base64:
                        result["image_base64"] = image_base64
                    results.append(result)
                else:
                    total_discarded += 1
                    if error_info:
                        error_info["sample_index"] = record.get("sample_index")
                        error_info["id"] = record.get("id")
                        error_info["source_a_id"] = source_a.get("id")
                        errors.append(error_info)
                if debug_output_dir is not None:
                    debug_records.append(debug_record)
                
                # 进度报告
                if not suppress_progress and total_processed % 10 == 0:
                    print(f"[进度] 已处理: {total_processed}, 成功: {len(results)}, 丢弃: {total_discarded}")
        
        # 保存成功结果
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存错误和丢弃的数据
        if errors:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_file = output_file.parent / f"{output_file.stem}_errors_{timestamp}.json"
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(errors, f, ensure_ascii=False, indent=2)
            print(f"  错误/丢弃数据已保存到: {error_file}")

        if debug_output_dir is not None and debug_records:
            debug_output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_file = debug_output_dir / f"{output_file.stem}_question_debug_{timestamp}.jsonl"
            with open(debug_file, "w", encoding="utf-8") as f:
                for item in debug_records:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"  问题生成调试记录已保存到: {debug_file}")
        
        if not suppress_progress:
            print(f"\n[完成] 处理完成！")
            print(f"  总处理: {total_processed}")
            print(f"  成功生成: {len(results)}")
            print(f"  丢弃/错误: {total_discarded}")
        print(f"  结果已保存到: {output_file}")
    
    def _extract_image_input(self, source_a: Dict[str, Any]) -> Optional[Any]:
        """
        从source_a中提取图片输入
        
        支持多种字段名：image_input, image_path, image_url等
        """
        # 尝试多种可能的字段名
        image_keys = [
            "image_input", "image_path", "image_url", "image",
            "img_input", "img_path", "img_url", "img"
        ]
        
        for key in image_keys:
            if key in source_a and source_a[key]:
                value = source_a[key]
                # 如果是字符串且看起来像路径或base64，返回
                if isinstance(value, str):
                    return value
                # 如果是其他类型（如bytes），也返回
                return value
        
        # 如果没有找到，返回None
        return None
    
    def _extract_image_base64(self, source_a: Dict[str, Any], image_input: Any) -> Optional[str]:
        """
        从source_a中提取图片的base64编码
        
        优先顺序：
        1. source_a中的image_base64字段
        2. 如果image_input是base64字符串，直接使用
        3. 其他可能的base64字段
        """
        # 优先从source_a中查找image_base64字段
        base64_keys = [
            "image_base64", "img_base64", "base64", "image_b64", "img_b64", "jpg"
        ]
        
        for key in base64_keys:
            if key in source_a and source_a[key]:
                value = source_a[key]
                if isinstance(value, str) and len(value) > 50:
                    # 移除可能的数据URL前缀
                    if value.startswith("data:image"):
                        match = re.search(r'base64,(.+)', value)
                        if match:
                            return match.group(1)
                        return value
                    return value
        
        # 如果 image_input 是本地路径，尝试读取并转 base64（支持 Step2 输出 image_path 而不内嵌 base64）
        if isinstance(image_input, str) and image_input:
            try:
                from pathlib import Path
                import base64

                p = Path(image_input)
                if p.is_file():
                    data = p.read_bytes()
                    if data:
                        return base64.b64encode(data).decode("ascii")
            except Exception:
                # 路径读取失败时，不阻断后续逻辑，继续尝试 base64 提取
                pass

        # 如果image_input是base64字符串，使用它
        if isinstance(image_input, str) and len(image_input) > 50:
            if image_input.startswith("data:image"):
                match = re.search(r'base64,(.+)', image_input)
                if match:
                    return match.group(1)
                return image_input
            # 简单验证：检查是否可能是base64
            base64_pattern = re.compile(r'^[A-Za-z0-9+/=]+$')
            if base64_pattern.match(image_input):
                return image_input
        
        return None
    
    def _extract_pipeline_from_record(self, record: Dict[str, Any]) -> Optional[str]:
        """
        从记录中提取pipeline类型
        
        支持多种字段名：pipeline_type, pipeline_name等
        """
        # 尝试多种可能的字段名
        pipeline_keys = ["pipeline_type", "pipeline_name", "pipeline"]
        
        for key in pipeline_keys:
            if key in record and record[key]:
                pipeline_name = record[key]
                # 验证pipeline是否存在
                available_pipelines = self.config_loader.list_pipelines()
                if pipeline_name in available_pipelines:
                    return pipeline_name
                
                # 尝试模糊匹配
                pipeline_config = self.config_loader.get_pipeline_config(pipeline_name)
                if pipeline_config:
                    config_name = pipeline_config.get("name", "")
                    if config_name == pipeline_name:
                        return pipeline_name
        
        # 尝试模糊匹配（基于关键词）
        pipeline_name_lower = str(record.get("pipeline_type", "")).lower()
        name_mapping = {
            "object counting": "object_counting",
            "object recognition": "question",
            "question": "question",
            "object position": "object_position",
            "object relative position": "object_relative_position",
            "relative position": "object_relative_position",
            "object proportion": "object_proportion",
            "object orientation": "object_orientation",
            "object absence": "object_absence",
            "place recognition": "place_recognition",
            "text association": "text_association",
            "caption": "caption"
        }
        
        available_pipelines = self.config_loader.list_pipelines()
        for key, pipeline_type in name_mapping.items():
            if key in pipeline_name_lower and pipeline_type in available_pipelines:
                return pipeline_type
        
        return None
    
    def _select_question_type(self) -> str:
        """
        根据配置的比例选择题型（复用原有逻辑）
        """
        rand = random.random()
        if rand < self.question_type_ratio["multiple_choice"]:
            return "multiple_choice"
        else:
            return "fill_in_blank"
    
    async def process_data_file_async(
        self,
        input_file: Path,
        output_file: Path,
        pipeline_names: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        num_gpus: int = 1,
        max_concurrent_per_gpu: int = 5,
        request_delay: float = 0.1,
        failed_selection_dir: Optional[Path] = None,
        debug_output_dir: Optional[Path] = None,
        suppress_progress: bool = False,
    ) -> None:
        """
        异步处理数据文件（使用预填充对象）
        """
        if failed_selection_dir is not None:
            self.failed_selection_dir = failed_selection_dir
            self.failed_selection_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] 读取输入文件: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"输入文件应该包含一个数组，但得到: {type(data)}")

        # 确定要使用的pipeline
        if pipeline_names is None:
            pipeline_names = self.config_loader.list_pipelines()

        print(f"[INFO] 使用pipelines: {pipeline_names}")
        print(f"[INFO] 总记录数: {len(data)}")

        if max_samples:
            data = data[:max_samples]
            print(f"[INFO] 限制处理前 {max_samples} 条记录")

        # 初始化异步客户端（按GPU数量创建）
        clients = []
        for gpu_id in range(max(1, num_gpus)):
            clients.append(
                AsyncGeminiClient(
                    gpu_id=gpu_id if num_gpus > 1 else None,
                    max_concurrent=max_concurrent_per_gpu,
                    request_delay=request_delay,
                    use_lb_client=False
                )
            )

        async def process_one(
            idx: int,
            record: Dict[str, Any],
            pipeline_name: str,
            async_client: AsyncGeminiClient
        ):
            """处理单条记录 + 单个pipeline"""
            fallback_events: List[Dict[str, Any]] = []
            debug_record = {
                "record_index": idx,
                "sample_index": record.get("sample_index"),
                "id": record.get("id"),
                "pipeline_name": pipeline_name,
                "prefill_input": record.get("prefill"),
                "input_record": record,
                "timestamp": datetime.now().isoformat(),
            }
            source_a = record.get("source_a", {})
            if not source_a:
                debug_record["error_stage"] = "data_loading"
                debug_record["error_reason"] = "记录没有source_a"
                return ("err", {
                    "record_index": idx,
                    "id": record.get("id"),
                    "error_stage": "data_loading",
                    "error_reason": "记录没有source_a",
                    "timestamp": datetime.now().isoformat()
                }, fallback_events, record, pipeline_name, debug_record)

            image_input = self._extract_image_input(source_a)
            image_base64 = self._extract_image_base64(source_a, image_input)
            if not image_base64:
                debug_record["error_stage"] = "data_loading"
                debug_record["error_reason"] = "无法提取image_base64"
                return ("err", {
                    "record_index": idx,
                    "id": record.get("id"),
                    "source_a_id": source_a.get("id"),
                    "error_stage": "data_loading",
                    "error_reason": "无法提取image_base64",
                    "timestamp": datetime.now().isoformat()
                }, fallback_events, record, pipeline_name, debug_record)

            prefill_input = record.get("prefill")
            if not prefill_input:
                debug_record["error_stage"] = "data_loading"
                debug_record["error_reason"] = "记录缺少prefill字段（需要包含'claim'或'target_object'）"
                return ("err", {
                    "record_index": idx,
                    "id": record.get("id"),
                    "error_stage": "data_loading",
                    "error_reason": "记录缺少prefill字段（需要包含'claim'或'target_object'）",
                    "timestamp": datetime.now().isoformat()
                }, fallback_events, record, pipeline_name, debug_record)

            # STEP 1: 加载Pipeline规范
            pipeline_config = self.config_loader.get_pipeline_config(pipeline_name)
            if not pipeline_config:
                debug_record["error_stage"] = "config_loading"
                debug_record["error_reason"] = f"Pipeline '{pipeline_name}' 不存在"
                return ("err", {
                    "record_index": idx,
                    "id": record.get("id"),
                    "error_stage": "config_loading",
                    "error_reason": f"Pipeline '{pipeline_name}' 不存在",
                    "timestamp": datetime.now().isoformat()
                }, fallback_events, record, pipeline_name, debug_record)

            # STEP 2: 处理预填充输入（简化版本）
            prefill_info = None
            try:
                # 检查是否使用新的简化格式（包含 prefilled_values）
                if "prefilled_values" in prefill_input:
                    prefill_info = await self.prefill_processor_simplified.process_prefill_async(
                        prefill_input=prefill_input,
                        image_base64=image_base64,
                        pipeline_config=pipeline_config,
                        async_client=async_client
                    )
                else:
                    # 向后兼容：处理旧格式（仅用于兼容可能存在的旧数据文件）
                    # 注意：新流程统一使用 claim + prefilled_values，不再使用 target_object
                    prefill_object = await self.prefill_processor.process_prefill_async(
                        prefill_input=prefill_input,
                        image_base64=image_base64,
                        pipeline_config=pipeline_config,
                        async_client=async_client
                    )
                    if prefill_object:
                        # 转换为新格式：PrefillProcessor 已经处理了优先级，直接使用 name
                        prefill_info = {
                            "claim": prefill_object.get("claim", ""),
                            "prefilled_values": {}
                        }
                        # PrefillProcessor 返回的 name 字段已经包含了正确的值（优先来自 target_object）
                        if prefill_object.get("name"):
                            prefill_info["prefilled_values"]["OBJECT_A"] = prefill_object["name"]
                        if prefill_object.get("other_object"):
                            prefill_info["prefilled_values"]["OBJECT_B"] = prefill_object["other_object"]
                
                if not prefill_info or not prefill_info.get("claim"):
                    debug_record["error_stage"] = "prefill_processing"
                    debug_record["error_reason"] = "预填充处理失败或缺少 claim"
                    return ("err", {
                        "record_index": idx,
                        "id": record.get("id"),
                        "error_stage": "prefill_processing",
                        "error_reason": "预填充处理失败或缺少 claim",
                        "timestamp": datetime.now().isoformat()
                    }, fallback_events, record, pipeline_name, debug_record)
                    
            except Exception as e:
                debug_record["error_stage"] = "prefill_processing"
                debug_record["error_reason"] = f"预填充处理过程出错: {str(e)}"
                return ("err", {
                    "record_index": idx,
                    "id": record.get("id"),
                    "error_stage": "prefill_processing",
                    "error_reason": f"预填充处理过程出错: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }, fallback_events, record, pipeline_name, debug_record)

            debug_record["prefill_info"] = prefill_info

            # STEP 3: 槽位填充（简化版本：使用 LLM 一次性填充所有 required_slots）
            claim = prefill_info.get("claim", "")
            prefilled_values = prefill_info.get("prefilled_values", {})
            required_slots = pipeline_config.get("required_slots", [])
            
            if required_slots:
                # 使用新的简化方法：LLM 一次性填充所有 required_slots
                slots = await self.slot_filler.fill_required_slots_with_llm_async(
                    image_base64=image_base64,
                    pipeline_config=pipeline_config,
                    claim=claim,
                    prefilled_values=prefilled_values,
                    async_client=async_client
                )
                if slots is None:
                    debug_record["error_stage"] = "slot_filling"
                    debug_record["error_reason"] = "槽位填充失败（LLM无法填充必需槽位）"
                    return ("err", {
                        "record_index": idx,
                        "id": record.get("id"),
                        "error_stage": "slot_filling",
                        "error_reason": "槽位填充失败（LLM无法填充必需槽位）",
                        "timestamp": datetime.now().isoformat()
                    }, fallback_events, record, pipeline_name, debug_record)
            else:
                slots = {}
            
            # 填充可选槽位（使用原有逻辑）
            optional_slots = pipeline_config.get("optional_slots", [])
            for slot in optional_slots:
                rand_val = random.random()
                if rand_val < 0.5:  # 50%概率填充可选槽位
                    value = await self.slot_filler._resolve_slot_async(
                        slot=slot,
                        image_base64=image_base64,
                        pipeline_config=pipeline_config,
                        selected_object=None,
                        is_optional=True,
                        async_client=async_client,
                        fallback_events=fallback_events
                    )
                    if value is not None:
                        slots[slot] = value

            debug_record["slots"] = slots
            # STEP 4: 问题生成
            question_type = self._select_question_type()
            debug_record["question_type"] = question_type
            # 构建 prefill_object 用于问题生成（向后兼容）
            prefill_object = {
                "claim": claim,
                "prefilled_values": prefilled_values,
                "source": "simplified"
            }
            question = await self.question_generator.generate_question_async(
                image_base64=image_base64,
                pipeline_config=pipeline_config,
                slots=slots,
                prefill_object=prefill_object,
                question_type=question_type,
                async_client=async_client
            )
            if not question:
                debug_record["error_stage"] = "question_generation"
                debug_record["error_reason"] = "问题生成失败（返回空）"
                return ("err", {
                    "record_index": idx,
                    "id": record.get("id"),
                    "error_stage": "question_generation",
                    "error_reason": "问题生成失败（返回空）",
                    "timestamp": datetime.now().isoformat()
                }, fallback_events, record, pipeline_name, debug_record)

            # STEP 5: 异步验证
            is_valid, reason = await self.validator.validate_async(
                question=question,
                image_base64=image_base64,
                pipeline_config=pipeline_config,
                global_constraints=self.global_constraints,
                async_client=async_client
            )
            if not is_valid:
                debug_record["error_stage"] = "question_validation"
                debug_record["error_reason"] = reason or "问题验证失败"
                return ("err", {
                    "record_index": idx,
                    "id": record.get("id"),
                    "error_stage": "question_validation",
                    "error_reason": reason or "问题验证失败",
                    "timestamp": datetime.now().isoformat()
                }, fallback_events, record, pipeline_name, debug_record)

            # STEP 6: 输出
            result = {
                "pipeline_name": pipeline_name,
                "pipeline_intent": pipeline_config.get("intent", ""),
                "question": question,
                "question_type": question_type,
                "answer_type": pipeline_config.get("answer_type", ""),
                "slots": slots,
                "prefill_object": prefill_object,
                "prefill_source": prefill_object.get("source"),
                "validation_reason": reason or "passed",
                "timestamp": datetime.now().isoformat(),
                "sample_index": record.get("sample_index"),
                "id": record.get("id"),
                "source_a_id": source_a.get("id"),
                "image_base64": image_base64
            }

            debug_record["question_type"] = question_type
            debug_record["prefill_object"] = prefill_object
            debug_record["slots"] = slots
            debug_record["question"] = question
            debug_record["error_stage"] = None
            debug_record["error_reason"] = None

            return ("ok", result, fallback_events, record, pipeline_name, debug_record)

        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        fallback_records: List[Dict[str, Any]] = []
        debug_records: List[Dict[str, Any]] = []
        total_processed = 0
        total_discarded = 0

        # 创建所有任务（按记录和pipeline）
        tasks = []
        for idx, record in enumerate(data, 1):
            record_pipeline = self._extract_pipeline_from_record(record)
            pipelines_to_use = [record_pipeline] if record_pipeline else pipeline_names
            for pipeline_name in pipelines_to_use:
                total_processed += 1
                client = clients[(total_processed - 1) % len(clients)]
                tasks.append(process_one(idx, record, pipeline_name, client))

        # 执行任务（并发由 AsyncGeminiClient 内部控制）
        # 使用 as_completed 实现进度报告
        completed = 0
        total = len(tasks)
        # 打开所有客户端
        async with AsyncExitStack() as stack:
            for client in clients:
                await stack.enter_async_context(client)
            for coro in asyncio.as_completed(tasks):
                tag, payload, fallback_events, record, pipeline_name, debug_record = await coro
                if fallback_events:
                    print(f"[WARNING] 触发兜底: record_index={record.get('sample_index')}, pipeline={pipeline_name}")
                    fallback_records.append({
                        "record": record,
                        "pipeline_name": pipeline_name,
                        "fallback_events": fallback_events,
                        "timestamp": datetime.now().isoformat()
                    })
                if debug_output_dir is not None:
                    debug_records.append(debug_record)
                if tag == "ok":
                    results.append(payload)
                else:
                    errors.append(payload)
                    total_discarded += 1
                completed += 1
                if not suppress_progress and (completed % 10 == 0 or completed == total):
                    print(f"[进度] 已处理: {completed}/{total}, 成功: {len(results)}, 丢弃: {total_discarded}")

        # 保存成功结果
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 保存错误和丢弃的数据
        if errors:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_file = output_file.parent / f"{output_file.stem}_errors_{timestamp}.json"
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(errors, f, ensure_ascii=False, indent=2)
            print(f"  错误/丢弃数据已保存到: {error_file}")

        if fallback_records:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fallback_file = output_file.parent / f"{output_file.stem}_fallback_{timestamp}.json"
            with open(fallback_file, 'w', encoding='utf-8') as f:
                json.dump(fallback_records, f, ensure_ascii=False, indent=2)
            print(f"  兜底样本已保存到: {fallback_file}")

        if debug_output_dir is not None and debug_records:
            debug_output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_file = debug_output_dir / f"{output_file.stem}_question_debug_{timestamp}.jsonl"
            with open(debug_file, "w", encoding="utf-8") as f:
                for item in debug_records:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"  问题生成调试记录已保存到: {debug_file}")

        if not suppress_progress:
            print(f"\n[完成] 处理完成！")
            print(f"  总处理: {total_processed}")
            print(f"  成功生成: {len(results)}")
            print(f"  丢弃/错误: {total_discarded}")
        print(f"  结果已保存到: {output_file}")
