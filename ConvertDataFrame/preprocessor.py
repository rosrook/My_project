"""
将VQA JSON格式（Base64图片）转换为代码2期望的格式
增强版：添加数据验证和清洗功能
"""
import json
import os
import base64
from pathlib import Path
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def decode_base64_image(image_base64):
    """解码Base64图片"""
    try:
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        return base64.b64decode(image_base64)
    except Exception as e:
        logger.warning(f"Base64解码失败: {e}")
        return None


def detect_image_format(img_bytes):
    """检测图片格式"""
    if not img_bytes or len(img_bytes) < 4:
        return 'jpg'
    
    # JPEG: FF D8 FF
    if img_bytes[:3] == b'\xff\xd8\xff':
        return 'jpg'
    # PNG: 89 50 4E 47
    elif img_bytes[:4] == b'\x89PNG':
        return 'png'
    
    return 'jpg'


def clean_text(text):
    """清理文本，移除空白字符"""
    if text is None:
        return ""
    return text.strip()


def build_conversation(vqa_item, item_index):
    """
    构建对话格式（增强版，带验证）
    
    Returns:
        (conversations, is_valid, error_msg)
    """
    # 获取问题
    full_question = vqa_item.get('full_question', vqa_item.get('question', ''))
    full_question = clean_text(full_question)
    
    if not full_question:
        return None, False, f"问题为空 (index {item_index})"
    
    # 获取答案
    answer = vqa_item.get('answer', '')
    answer = clean_text(answer)
    
    if not answer:
        return None, False, f"答案为空 (index {item_index})"
    
    # 获取其他信息
    explanation = vqa_item.get('explanation', '')
    explanation = clean_text(explanation)
    options = vqa_item.get('options', {})
    
    # 处理答案（如果是选项字母，获取完整选项文本）
    if answer and options and isinstance(options, dict):
        answer_upper = answer.upper().strip()
        if len(answer_upper) == 1 and answer_upper in options:
            option_text = clean_text(options[answer_upper])
            if option_text:
                answer_text = f"{answer_upper}. {option_text}"
            else:
                answer_text = answer
        else:
            answer_text = answer
    else:
        answer_text = answer
    
    # 构建助手消息
    if explanation:
        assistant_content = f"{answer_text}\n\nExplanation: {explanation}"
    else:
        assistant_content = answer_text
    
    assistant_content = clean_text(assistant_content)
    
    # 最终验证
    if not assistant_content:
        return None, False, f"助手回复为空 (index {item_index})"
    
    # 返回对话格式（使用 from/value 格式，代码2会转换为 role/content）
    conversations = [
        {
            'from': 'human',
            'value': full_question
        },
        {
            'from': 'gpt',
            'value': assistant_content
        }
    ]
    
    return conversations, True, None


def validate_conversation(conversations, item_index):
    """
    验证对话格式
    
    Returns:
        (is_valid, error_msg)
    """
    if not conversations or not isinstance(conversations, list):
        return False, f"对话不是列表格式 (index {item_index})"
    
    if len(conversations) < 2:
        return False, f"对话少于2轮 (index {item_index})"
    
    for i, msg in enumerate(conversations):
        if not isinstance(msg, dict):
            return False, f"消息 {i} 不是字典格式 (index {item_index})"
        
        if 'from' not in msg or 'value' not in msg:
            return False, f"消息 {i} 缺少 from 或 value 字段 (index {item_index})"
        
        if msg['from'] not in ['human', 'gpt', 'system']:
            return False, f"消息 {i} 的 from 字段无效: {msg['from']} (index {item_index})"
        
        content = msg.get('value', '')
        if not content or not clean_text(content):
            return False, f"消息 {i} 的内容为空 (index {item_index})"
    
    # 检查第一条必须是 human
    if conversations[0]['from'] != 'human':
        return False, f"第一条消息不是 human (index {item_index})"
    
    # 检查第二条必须是 gpt
    if conversations[1]['from'] != 'gpt':
        return False, f"第二条消息不是 gpt (index {item_index})"
    
    return True, None


def convert_vqa_to_standard_format(vqa_json_path, output_dir, strict_mode=True):
    """
    转换VQA JSON为代码2期望的格式
    
    Args:
        vqa_json_path: VQA JSON文件路径
        output_dir: 输出目录
        strict_mode: 严格模式，遇到错误跳过该样本
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取VQA数据
    logger.info(f"读取文件: {vqa_json_path}")
    with open(vqa_json_path, 'r', encoding='utf-8') as f:
        vqa_data = json.load(f)
    
    if not isinstance(vqa_data, list):
        logger.error(f"错误: VQA JSON应该是列表格式")
        return
    
    logger.info(f"开始转换 {len(vqa_data)} 条数据...")
    
    success_count = 0
    skip_count = 0
    error_stats = {
        'no_image': 0,
        'image_decode_fail': 0,
        'empty_question': 0,
        'empty_answer': 0,
        'conversation_invalid': 0,
        'other': 0
    }
    
    # 创建错误日志文件
    error_log_path = output_dir / 'conversion_errors.log'
    error_log = open(error_log_path, 'w', encoding='utf-8')
    
    for idx, vqa_item in enumerate(tqdm(vqa_data, desc="转换进度")):
        try:
            # 1. 解码图片
            image_base64 = vqa_item.get('image_base64')
            if not image_base64:
                error_stats['no_image'] += 1
                error_log.write(f"[{idx}] 跳过: 缺少image_base64\n")
                skip_count += 1
                continue
            
            img_bytes = decode_base64_image(image_base64)
            if img_bytes is None:
                error_stats['image_decode_fail'] += 1
                error_log.write(f"[{idx}] 跳过: 图片解码失败\n")
                skip_count += 1
                continue
            
            # 2. 构建对话
            conversations, is_valid, error_msg = build_conversation(vqa_item, idx)
            
            if not is_valid:
                if 'question' in error_msg or '问题' in error_msg:
                    error_stats['empty_question'] += 1
                elif 'answer' in error_msg or '答案' in error_msg:
                    error_stats['empty_answer'] += 1
                else:
                    error_stats['conversation_invalid'] += 1
                
                error_log.write(f"[{idx}] 跳过: {error_msg}\n")
                skip_count += 1
                continue
            
            # 3. 验证对话格式
            is_valid, error_msg = validate_conversation(conversations, idx)
            if not is_valid:
                error_stats['conversation_invalid'] += 1
                error_log.write(f"[{idx}] 跳过: {error_msg}\n")
                skip_count += 1
                continue
            
            # 4. 检测图片格式
            img_format = detect_image_format(img_bytes)
            
            # 5. 保存图片文件
            img_filename = f"image_{idx:06d}.{img_format}"
            img_path = output_dir / img_filename
            with open(img_path, 'wb') as f:
                f.write(img_bytes)
            
            # 6. 构建JSON数据（代码2期望的格式）
            json_data = {
                'image_url': str(img_path),
                'conversations': conversations
            }
            
            # 7. 保存JSON文件
            json_filename = f"image_{idx:06d}.json"
            json_path = output_dir / json_filename
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            success_count += 1
            
        except Exception as e:
            error_stats['other'] += 1
            error_log.write(f"[{idx}] 异常: {str(e)}\n")
            logger.error(f"处理索引 {idx} 时出错: {e}")
            skip_count += 1
            continue
    
    error_log.close()
    
    # 打印统计信息
    logger.info(f"\n{'='*60}")
    logger.info(f"转换完成!")
    logger.info(f"{'='*60}")
    logger.info(f"成功: {success_count}/{len(vqa_data)} ({success_count/len(vqa_data)*100:.2f}%)")
    logger.info(f"跳过: {skip_count}/{len(vqa_data)} ({skip_count/len(vqa_data)*100:.2f}%)")
    logger.info(f"\n错误统计:")
    logger.info(f"  - 缺少图片: {error_stats['no_image']}")
    logger.info(f"  - 图片解码失败: {error_stats['image_decode_fail']}")
    logger.info(f"  - 问题为空: {error_stats['empty_question']}")
    logger.info(f"  - 答案为空: {error_stats['empty_answer']}")
    logger.info(f"  - 对话格式错误: {error_stats['conversation_invalid']}")
    logger.info(f"  - 其他错误: {error_stats['other']}")
    logger.info(f"\n输出目录: {output_dir}")
    logger.info(f"错误日志: {error_log_path}")


def batch_convert(vqa_json_files, output_base_dir):
    """
    批量转换多个VQA JSON文件
    
    Args:
        vqa_json_files: VQA JSON文件列表
        output_base_dir: 输出基础目录
    """
    output_base_dir = Path(output_base_dir)
    
    for vqa_file in vqa_json_files:
        vqa_file = Path(vqa_file)
        logger.info(f"\n处理文件: {vqa_file.name}")
        
        # 为每个文件创建子目录
        output_dir = output_base_dir / vqa_file.stem
        convert_vqa_to_standard_format(str(vqa_file), str(output_dir))


def diagnose_vqa_file(vqa_json_path, max_check=100):
    """
    诊断VQA文件，查找问题数据
    
    Args:
        vqa_json_path: VQA JSON文件路径
        max_check: 最多检查的样本数
    """
    logger.info(f"诊断文件: {vqa_json_path}")
    
    with open(vqa_json_path, 'r', encoding='utf-8') as f:
        vqa_data = json.load(f)
    
    if not isinstance(vqa_data, list):
        logger.error("文件不是列表格式")
        return
    
    logger.info(f"总样本数: {len(vqa_data)}")
    
    problem_samples = []
    
    for idx, item in enumerate(vqa_data[:max_check]):
        issues = []
        
        # 检查必需字段
        if 'image_base64' not in item:
            issues.append("缺少 image_base64")
        
        if 'full_question' not in item and 'question' not in item:
            issues.append("缺少 question/full_question")
        
        if 'answer' not in item:
            issues.append("缺少 answer")
        
        # 检查内容是否为空
        question = item.get('full_question', item.get('question', ''))
        if not clean_text(question):
            issues.append("问题为空")
        
        answer = item.get('answer', '')
        if not clean_text(answer):
            issues.append("答案为空")
        
        if issues:
            problem_samples.append({
                'index': idx,
                'issues': issues,
                'sample': item
            })
    
    if problem_samples:
        logger.warning(f"\n发现 {len(problem_samples)} 个问题样本:")
        for ps in problem_samples[:10]:  # 只显示前10个
            logger.warning(f"  索引 {ps['index']}: {', '.join(ps['issues'])}")
        
        # 保存问题样本
        problem_file = Path(vqa_json_path).parent / 'problem_samples.json'
        with open(problem_file, 'w', encoding='utf-8') as f:
            json.dump(problem_samples, f, ensure_ascii=False, indent=2)
        logger.info(f"问题样本已保存到: {problem_file}")
    else:
        logger.info("未发现问题样本")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='转换VQA JSON为标准格式')
    parser.add_argument('--input', type=str, required=True, 
                       help='VQA JSON文件路径或包含多个文件的目录')
    parser.add_argument('--output', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--diagnose', action='store_true',
                       help='仅诊断文件，不转换')
    parser.add_argument('--max_check', type=int, default=1000,
                       help='诊断时最多检查的样本数')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if args.diagnose:
        # 诊断模式
        if input_path.is_file():
            diagnose_vqa_file(str(input_path), args.max_check)
        else:
            logger.error("诊断模式仅支持单个文件")
    else:
        # 转换模式
        if input_path.is_file():
            convert_vqa_to_standard_format(str(input_path), args.output)
        elif input_path.is_dir():
            vqa_files = list(input_path.glob('vqa_dataset_successful_*.json'))
            if not vqa_files:
                vqa_files = list(input_path.glob('*.json'))
            
            if vqa_files:
                batch_convert(vqa_files, args.output)
            else:
                logger.error(f"在 {input_path} 中未找到JSON文件")
        else:
            logger.error(f"错误: {input_path} 不存在")
