"""
项目配置文件
"""
import os
from pathlib import Path

# 尝试加载环境变量（dotenv为可选依赖）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # 如果没有安装dotenv，直接从环境变量读取
    pass

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# # API配置（使用OpenAI兼容格式）
# API_KEY = os.getenv("API_KEY", "sk")  # 在 QS 平台生成的 token
# BASE_URL = os.getenv("BASE_URL", "http://10.158.146.63:8081/v1")  # DirectLLM 域名
# MODEL_NAME = os.getenv("MODEL_NAME", "/workspace/models/Qwen3-VL-235B-A22B-Instruct/")  # 模型名称，在 Body 中指明要访问的模型名

# -----------------
# API配置（使用LBOpenAIClient）
SERVICE_NAME = os.getenv("SERVICE_NAME", "mediak8s-editprompt-qwen235b")  # 服务名称
ENV = os.getenv("ENV", "prod")  # 环境（prod/staging等）
API_KEY = os.getenv("API_KEY", "1")  # API密钥（LBOpenAIClient需要，但可能不使用）
MODEL_NAME = os.getenv("MODEL_NAME", "/workspace/Qwen3-VL-235B-A22B-Instruct")  # 模型名称

# 向后兼容配置（已废弃，建议使用上面的配置）
BASE_URL = os.getenv("BASE_URL", "https://maas.devops.xiaohongshu.com/v1")  # DirectLLM 域名（仅用于兼容）
# ------------------

# 向后兼容（已废弃，建议使用上面的配置）
GEMINI_API_KEY = API_KEY
GEMINI_MODEL = MODEL_NAME

# Pipeline配置
PIPELINE_CONFIG = {
    "question": {
        "name": "Question Pipeline",
        "description": "Concept grounding and term matching",
        "question": "Which term best matches the concept conveyed by the image?",
        "criteria": [
            "The image conveys a clear and identifiable concept, category, or property that can be named by a single term",
            "The concept is intrinsic to the depicted content, rather than dependent on superficial visual appearance",
            "The concept can be inferred from the image content without relying on external textual context",
            "[optional] Multiple objects are present but collectively support the same underlying concept",
            "[optional] The image contains a strong semantic anchor that explicitly reinforces the target concept",
            "[optional] The concept is abstract, relational, or attribute-based rather than a concrete object label",
            "[optional] The concept is not immediately obvious and requires non-trivial semantic inference"
        ]
    },

    "caption": {
        "name": "Caption Pipeline",
        "description": "Scene description/caption",
        "question": "Which one is the correct caption of this image?",
        "criteria": [
            "Image depicts a real-world photographic scene (not illustration, diagram, or synthetic image)",
            "Multiple objects of different semantic types are present",
            "Objects have clearly distinguishable spatial positions and relationships",
            "Objects are contextually consistent with the background environment",
            "Most objects are largely complete and not heavily occluded",
            "[optional] Empty or background-only regions do not exceed approximately one-third of the image"
        ]
    },

    "place_recognition": {
        "name": "Place Recognition Pipeline",
        "description": "Geographic location identification",
        "question": "What is the name of the place shown?",
        "criteria": [
            "Image depicts a real-world, non-fictional geographic location represented in map form",
            "The map contains a visually salient and deliberately highlighted region that can plausibly serve as the target location",
            "The highlighted region is perceptually distinct from the surrounding map context",
            "The map provides sufficient geographic cues—primarily the shape of the highlighted region and its surrounding spatial context—to support location inference, even when the place name cannot be directly obtained from the image alone",
            "[optional] Identifying the location requires external geographic knowledge beyond what is explicitly visible in the image",
            "[optional] The map contains minimal textual labels or place names, avoiding trivial identification through reading",
            "[optional] The map may exhibit mild geometric or scale distortion that increases difficulty without obscuring essential geographic structure"
        ]
    },

    "text_association": {
        "name": "Text Association Pipeline",
        "description": "Basic image-to-text alignment",
        "question": "Which text best describes what is shown in the image?",
        "criteria": [
            "The image contains visually perceivable content that can be reasonably described in natural language",
            "One or more objects, scenes, or activities are present and can serve as the basis for a textual description",
            "The image does not rely on extensive external context to be meaningfully described",
            "[optional] The image has a loosely identifiable main subject or scene",
            "[optional] The image supports a generally consistent description across different viewers, even if interpretations vary in detail",
            "[optional] The image is not severely abstract, noisy, or visually degraded to the extent that description becomes unreliable"
        ]
    },

    "object_proportion": {
        "name": "Object Proportion Pipeline",
        "description": "Object size proportion in image",
        "question": "Approximately what proportion of the picture is occupied by [potential target]?",
        "criteria": [
            "Image contains at least one visually salient object category that can plausibly serve as a potential target, even if not explicitly specified in the query",
            "Potential target object is complete or sufficiently recognizable to allow approximate size estimation",
            "The image is not overwhelmingly dominated by a single potential target object covering nearly the entire frame",
            "Potential target object boundaries are visually discernible for proportion assessment",
            "Potential target object category is unambiguous and conceptually clear",
            "[optional] The potential target object may be absent from the image, requiring a zero-proportion judgment",
            "[optional] The image contains multiple instances of the potential target object or complex arrangements that increase proportion estimation difficulty",
            "[optional] The potential target object occupies a small or partial region, making size estimation more challenging"
        ]
    },

    "object_position": {
        "name": "Object Position Pipeline",
        "description": "Object location in image",
        "question": "Where is the [object] located in the picture?",
        "criteria": [
            "Image contains at least one visually salient object that can serve as a potential target",
            "At least one such object has visually identifiable boundaries or distinguishable parts",
            "At least one such object can be treated as a single coherent entity for localization",
            "[optional] Image contains visually similar objects that may act as distractors",
            "[optional] Potential target object occupies a relatively small region of the image",
            "[optional] Potential target object is partially visible (e.g., body parts, silhouette, color patch)",
            "[optional] Potential target object appears in challenging regions (shadows, background, corners, or partial occlusion)"
        ]
    },

    "object_absence": {
        "name": "Object Absence Pipeline",
        "description": "Identifying areas without certain objects",
        "question": "Which corner doesn't have any [objects]?",
        "criteria": [
            "Image contains at least one visually salient object category that can serve as a potential target, even if not explicitly specified in the query",
            "At least one type of potential target objects in the image has visually clear and contiguous boundaries",
            "Potential target objects occupy a sufficiently large and contiguous area of the image, rather than appearing as small or isolated instances",
            "When the image is partitioned into predefined spatial regions (e.g., four corners), potential target objects are present in at least two regions",
            "At most two spatial regions may be completely absent of potential target objects"
        ]
    },

    "object_orientation": {
        "name": "Object Orientation Pipeline",
        "description": "Object facing direction",
        "question": "In the picture, which direction is this [object] facing?",
        "criteria": [
            "Image contains at least one visually salient object that can serve as a target instance, even if the query does not explicitly specify the object",
            "The potential target object belongs to a category that is inherently directional, rather than orientation-neutral, regardless of whether the direction is visually salient",
            "[optional] Target object occupies a small portion of the image",
            "[optional] Target object is partially occluded or visually blurred",
            "[optional] Image contains interacting or overlapping objects requiring disambiguation",
            "[optional] Orientation judgment requires fine-grained directional discrimination"
        ]
    },

    "object_counting": {
        "name": "Object Counting Pipeline",
        "description": "Counting objects in image",
        "question": "How many [objects] are in the picture?",
        "criteria": [
            "Image contains at least one visually identifiable object category that can be counted, even if the target category is not explicitly specified in the query",
            "Target objects are countable as discrete instances",
            "[optional] Target objects are partially visible or truncated",
            "[optional] Image contains visually similar distractor objects",
            "[optional] Target objects overlap or occlude each other",
            "[optional] Target objects appear under unusual poses, rotations, or viewpoints",
            "[optional] Target objects visually blend with background colors or textures",
            "[optional] Image contains zero instances of the target object",
            "[optional] Multiple visually similar object subtypes may need to be jointly counted",
            "[optional] Object appearance varies due to packaging, surface decoration, or deformation",
            "[optional] Image exhibits challenging visual conditions (low light, blur, reflections)",
            "[optional] Mirrors or reflections introduce duplicated or misleading object appearances"
        ]
    }
}

# 数据路径配置
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

# 数据匹配配置
RECAT_ROOT = os.getenv("RECAT_ROOT", "/home/zhuxuzhou/recat")  # 重分类后的根目录
BENCHMARK_FILE = os.getenv("BENCHMARK_FILE", "/mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v1.5/bench/MMBench_DEV_EN_V11/MMBench_DEV_EN_V11.parquet")  # 基准parquet文件路径
MATCH_OUTPUT_DIR = os.getenv("MATCH_OUTPUT_DIR", str(DATA_DIR / "matched_output"))  # 匹配结果输出目录

# 数据匹配测试模式配置
MATCH_TEST_MODE = os.getenv("MATCH_TEST_MODE", "False").lower() == "false"  # 测试模式
MATCH_TEST_SAMPLES = int(os.getenv("MATCH_TEST_SAMPLES", "5"))  # 测试模式下每个类别处理的样本数
MATCH_TEST_MAX_CATEGORIES = int(os.getenv("MATCH_TEST_MAX_CATEGORIES", "2"))  # 测试模式下最多处理的类别数

# 创建必要的目录
DATA_DIR.mkdir(exist_ok=True)
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
Path(MATCH_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
