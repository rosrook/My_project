""" Convert Parquet dataset into WebDataset (WDS) format - FIXED VERSION """
import argparse
import json
import os
import yaml
import webdataset as wds
from tqdm import tqdm
import random
import pyarrow.parquet as pq
from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseWebdatasetFactory
from megatron.energon.flavors.webdataset import MAIN_FOLDER_NAME
from glob import glob
import logging
import pandas as pd
from multiprocessing import Pool, Process, Queue, Manager
import queue
from io import BytesIO
from PIL import Image
import re

def sample_loader_template(media: str=None):
    """Returns a template for a sample_loader.py file."""
    # 根据媒体类型决定返回哪些字段
    if media == 'video':
        return_fields = "        video=video if len(video) > 0 else None,"
    elif media == 'image':
        return_fields = "        image=image if len(image) > 0 else None,"
    else:  # mix 或其他
        return_fields = "        video=video if len(video) > 0 else None,\n        image=image if len(image) > 0 else None,"
    
    return "\n".join([
        "def sample_loader(sample: dict) -> dict:",
        "    messages=[]",
        "    system=None",
        "    for message in sample['json']['texts']:",
        "        assert message['role'] in ['system','user','assistant']",
        "        if message['role'] == 'system':",
        "            system=message['content']",
        "            continue",
        "        messages.append(dict(",
        "            role=message['role'],",
        "            content=message['content']",
        "        ))",
        "    video = []",
        "    image = []",
        "    if sample['json']['media'] == 'video':",
        "        for name in sample['json']['name']:",
        "            video.append(sample.get(name))",
        "    elif sample['json']['media'] == 'image':",
        "        for name in sample['json']['name']:",
        "            image.append(sample.get(name))",
        "    return dict(",
        "        __key__=sample['__key__'],",
        "        __restore_key__=sample['__restore_key__'],",
        return_fields,
        "        system=system,",
        "        messages=messages,",
        "    )",
        "def part_filter(part: str) -> bool:",
        "    return True",
    ])

def apply_template(texts, num_img):
    """FIXED: 创建副本避免修改原始数据"""
    new_text = []
    for it, text in enumerate(texts):
        # 创建副本
        text_copy = text.copy()
        
        # 跳过 system 消息
        if text_copy.get('from', None) == 'system':
            continue
        
        # 转换 from -> role
        if text_copy.get('from') == 'user' or text_copy.get('from') == 'human':
            text_copy['role'] = 'user'
            if 'from' in text_copy:
                text_copy.pop('from')
        elif text_copy.get('from') == 'gpt' or text_copy.get('from') == 'assistant':
            text_copy['role'] = 'assistant'
            if 'from' in text_copy:
                text_copy.pop('from')
        
        # 转换 value -> content
        if text_copy.get('value') is not None:
            text_copy['content'] = text_copy.pop('value')
        
        # 确保 content 字段存在且不为 None
        if 'content' not in text_copy:
            text_copy['content'] = ''
        
        # 确保 content 是字符串
        if text_copy['content'] is None:
            text_copy['content'] = ''
        
        # 添加 <image> 标记
        if it == 0:
            if '<image>' not in text_copy['content'] and num_img > 0:
                imgstr = ['<image>'] * num_img
                if text_copy['content'].startswith('\n'):
                    text_copy['content'] = text_copy['content'].lstrip('\n')
                text_copy['content'] = ''.join(imgstr) + '\n' + text_copy['content']
        
        new_text.append(text_copy)
    
    return new_text

def apply_question_answer_template(question, answer):
    question = question.strip()
    conv = [
        {'role': 'user', 'content': question},
        {'role': 'assistant', 'content': answer.strip()},
    ]
    return conv

def check_conversation_format(conv):
    """FIXED: 增强的检查函数，带详细日志"""
    if not isinstance(conv, list):
        logging.warning(f"对话不是列表格式: {type(conv)}")
        return False
    
    allgood = True
    for ii, turn in enumerate(conv):
        # 检查是否是字典
        if not isinstance(turn, dict):
            logging.warning(f"消息 {ii} 不是字典: {type(turn)}")
            allgood = False
            continue
        
        role = turn.get('role', '')
        content = turn.get('content', '')
        
        # 检查 role
        if role not in ['user', 'assistant']:
            logging.warning(f"消息 {ii} role 错误: '{role}'")
            allgood = False
        
        # 检查 content（增强检查）
        if content is None:
            logging.warning(f"消息 {ii} content 是 None")
            allgood = False
        elif not isinstance(content, str):
            logging.warning(f"消息 {ii} content 不是字符串: {type(content)}")
            allgood = False
        elif len(content.strip()) < 1:
            logging.warning(f"消息 {ii} content 为空或只有空白字符")
            allgood = False
    
    return allgood

def construct_sample_from_row(row, index, media_type, media_bytes, args):
    """从 Parquet 行构建 WebDataset sample"""
    vision_data = {}
    vision_name = []

    if media_type in ['image', 'video']:
        media_key = 'image' if media_type == 'image' else 'video'
        
        if isinstance(media_bytes, bytes):
            vision_data[f"0_{media_key}"] = media_bytes
            vision_name.append(f"0_{media_key}")
        elif isinstance(media_bytes, dict) and 'bytes' in media_bytes:
            vision_data[f"0_{media_key}"] = media_bytes['bytes']
            vision_name.append(f"0_{media_key}")
        elif isinstance(media_bytes, list):
            for i, media_item in enumerate(media_bytes):
                if isinstance(media_item, bytes):
                    vision_data[f"{i}_{media_key}"] = media_item
                    vision_name.append(f"{i}_{media_key}")
                elif isinstance(media_item, dict) and 'bytes' in media_item:
                    vision_data[f"{i}_{media_key}"] = media_item['bytes']
                    vision_name.append(f"{i}_{media_key}")
        else:
            logging.warning(f"未知的媒体数据格式，跳过{row.get('__source_file__', '')} index {index}")
            return None

    conv = row.get(args.columns_messages, row.get('messages', row.get('texts')))
    
    if isinstance(conv, dict):
        conv = apply_question_answer_template(conv.get('question', ''), conv.get('answer', ''))

    conv = apply_template(conv, len(vision_name)) if conv is not None else None
    
    if conv is None or len(conv) < 2:
        logging.warning(f"对话为空或少于2轮，跳过{row.get('__source_file__', '')} index {index}")
        return None
    
    if conv[0].get('role', None) != 'user' or conv[1].get('role', None) != 'assistant':
        logging.warning(f"对话角色顺序错误，跳过{row.get('__source_file__', '')} index {index}")
        return None
    
    content = {
        "texts": conv,
        "media": media_type,
        "name": vision_name if vision_name else None
    }

    sample = {
        "__source_file__": row.get('__source_file__', ''),
        "__key__": f"{media_type}_{index}",
        **vision_data,
        "json": json.dumps(content, ensure_ascii=False).encode("utf-8"),
    }

    # 验证媒体数据
    for key, val in vision_data.items():
        if not isinstance(val, bytes):
            logging.warning(f"媒体数据不是字节类型，跳过{row.get('__source_file__', '')} index {index}")
            return None

    return sample

def parallel_file_reader(file_path_list, batch_size):
    for file_path in file_path_list:
        """单个文件的读取进程"""
        try:
            json_path = file_path
            with open(json_path, 'r', encoding='utf-8') as f:
                json_dat = json.load(f)
            
            img_path = json_dat.get('image_url', None)
            if img_path is None:
                img_path = os.path.splitext(file_path)[0] + '.jpg'

            img_bytes = open(img_path, 'rb').read()

            row_dict = {
                'image': img_bytes,
                'conversations': json_dat.get('conversations', json_dat)
            }
            row_dict['__source_file__'] = file_path
            output_queue.put(row_dict)
        except Exception as e:
            logging.error(f"读取文件 {file_path} 出错: {e}")

    output_queue.put(None)

def parallel_multi_file_iterator(file_paths, num_workers=4, shuffle_buffer_size=100000):
    processes = []
    files_per_worker = len(file_paths) // num_workers

    for i in range(num_workers):
        start_idx = i * files_per_worker
        end_idx = start_idx + files_per_worker if i < num_workers - 1 else len(file_paths)
        worker_files = file_paths[start_idx:end_idx]

        p = Process(target=parallel_file_reader, args=(worker_files, batch_size))
        p.start()
        processes.append(p)

    shuffle_buffer = []
    finished_count = 0

    while finished_count < num_workers or shuffle_buffer:
        while len(shuffle_buffer) < shuffle_buffer_size and finished_count < num_workers:
            try:
                item = output_queue.get(timeout=0.1)
                if item is None:
                    finished_count += 1
                else:
                    shuffle_buffer.append(item)
            except queue.Empty:
                break

        if shuffle_buffer:
            idx = random.randint(0, len(shuffle_buffer) - 1)
            yield shuffle_buffer.pop(idx)

    for p in processes:
        p.join()

def convert_parquet_to_wds(args):
    """将 Parquet 数据集转换为 WDS 格式"""
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if isinstance(args.data_root, list):
        parquet_files = []
        for path in args.data_root:
            if os.path.isdir(path):
                json_files = sorted(glob(os.path.join(path, "*.json")))
                if not json_files:
                    json_files = sorted(glob(os.path.join(path, "**/*.json"), recursive=True))
                parquet_files.extend(json_files)

    print(f"找到 {len(parquet_files)} 个 JSON 文件")
    random.shuffle(parquet_files)

    if args.debug_file is not None and args.debug_file > 0:
        parquet_files = parquet_files[:args.debug_file]
        print(f"调试模式，仅处理前 {args.debug_file} 个文件")

    data_iterator = parallel_multi_file_iterator(
        file_paths=parquet_files,
        num_workers=args.num_workers
    )

    tar = os.path.join(args.output_dir, 'subtaskdata-%d.tar')
    print(f"开始写入 WebDataset 到 {args.output_dir}")

    success_count = 0
    skip_count = 0

    with wds.ShardWriter(tar, maxcount=args.maxcount, maxsize=args.maxsize) as shard_writer:
        for index, row in enumerate(tqdm(data_iterator, desc="Converting to WDS")):
            try:
                sample = construct_sample_from_row(row, index, 'image', row.get('image', None), args)
                
                if sample is None:
                    skip_count += 1
                    continue
                
                jsondat = json.loads(sample.get('json', None).decode('utf-8'))
                conv = jsondat.get('texts', None)
                
                # 验证对话
                if not check_conversation_format(conv):
                    logging.warning(f"对话内容格式错误，跳过{row.get('__source_file__', '')} index {index}")
                    skip_count += 1
                    continue
                
                if len(conv) < 2:
                    logging.warning(f"对话轮次不足，跳过{row.get('__source_file__', '')} index {index}")
                    skip_count += 1
                    continue
                
                if '<image>' not in conv[0].get('content', '') and jsondat.get('media', None) in ['image', 'video', 'mix']:
                    logging.warning(f"缺少图片标记，跳过{row.get('__source_file__', '')} index {index}")
                    skip_count += 1
                    continue

                # 验证媒体数据
                allgood = True
                if jsondat['media'] == 'video':
                    for name in jsondat['name']:
                        if not isinstance(sample.get(name), bytes):
                            allgood = False
                elif jsondat['media'] == 'image':
                    for name in jsondat['name']:
                        if not isinstance(sample.get(name), bytes):
                            allgood = False
                else:
                    skip_count += 1
                    continue
                
                if not allgood:
                    logging.warning(f"媒体数据验证失败{row.get('__source_file__', '')} {index}")
                    skip_count += 1
                    continue

                # 检查媒体数量
                num_img2 = 0
                num_img = len(jsondat['name'])
                for key in sample.keys():
                    if '_image' in key:
                        num_img2 += 1
                
                if num_img != num_img2:
                    logging.warning(f"图片数量不匹配: expected {num_img}, found {num_img2}")
                    skip_count += 1
                    continue
                
                # 检查标记数量
                media = jsondat.get('media', '')
                content_text = conv[0].get('content', '')
                img_in_content = re.findall(r'<{}>'.format(media), content_text)
                num_img_in_content = len(img_in_content)
                
                if num_img != num_img_in_content:
                    logging.warning(f"标记数量不匹配: expected {num_img}, found {num_img_in_content}")
                    skip_count += 1
                    continue

                shard_writer.write(sample)
                success_count += 1
                
            except Exception as e:
                logging.error(f"处理第 {row.get('__source_file__', '')}, {index} 行时出错: {e}")
                import traceback
                traceback.print_exc()
                skip_count += 1
                continue

    print(f"\n转换完成: 成功 {success_count}, 跳过 {skip_count}")

    if args.media in ["mix", "video", "image"]:
        write_config(EPath(args.output_dir).absolute(), args.media)

    print(f"数据集成功转换为 WebDataset 格式")

def write_config(path: EPath, media: str=None):
    """写入配置到指定路径"""
    (path / MAIN_FOLDER_NAME).mkdir(exist_ok=True)
    all_tars = list(path.glob("**/*.tar")) + list(path.glob("**/*.tgz"))
    all_tars = [str(p.relative_to(path)) for p in sorted(all_tars)]

    # 修复：根据 media 类型选择正确的样本类
    if media == 'mix':
        class_type = "MultiMixQASample"
    elif media == 'video':
        class_type = "MultiVidQASample"
    elif media == 'image':
        class_type = "MultiMixQASample"  # 图像也使用 MultiMixQASample
    else:
        class_type = "MultiMixQASample"  # 默认使用 MultiMixQASample
    
    dataset_definition = {
        "sample_type": {
            "__module__": "aiak_training_llm.data.multimodal",
            "__class__": class_type,
        },
        "part_filter": "sample_loader.py:part_filter",
        "sample_loader": "sample_loader.py:sample_loader"
    }

    with (path / MAIN_FOLDER_NAME / "dataset.yaml").open("w") as f:
        yaml.dump(dataset_definition, f, sort_keys=False)

    with (path / MAIN_FOLDER_NAME / "sample_loader.py").open("w") as f:
        f.write(sample_loader_template(media))

    BaseWebdatasetFactory.prepare_dataset(
        path,
        all_tars,
        split_parts_ratio=[("train", 1.0), ("val", 0), ("test", 0)],
        tar_index_only=False,
        workers=96,
    )

def parse_args():
    """解析参数"""
    parser = argparse.ArgumentParser(description='Convert JSON dataset to WebDataset format')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--data_root', nargs='+', required=True, help='JSON 文件路径或目录')
    parser.add_argument('--maxcount', type=int, default=128, help='每个 shard 的样本数')
    parser.add_argument('--maxsize', type=int, default=3000000000, help='每个 shard 的最大大小')
    parser.add_argument('--media', type=str, choices=["mix", "image", "video"], default="image", help='媒体类型')
    parser.add_argument('--columns_messages', type=str, default="conversations", help='消息列名')
    parser.add_argument('--shuffle', action='store_true', help='是否 shuffle 数据')
    parser.add_argument('--batch_size', type=int, default=20, help='每批读取的行数')
    parser.add_argument('--shuffle_buffer_size', type=int, default=100000, help='Shuffle 缓冲区大小')
    parser.add_argument('--num_workers', type=int, default=16, help='并行读取的工作进程数')
    parser.add_argument('--debug_file', type=int, default=-1, help='调试模式文件数')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    manager = Manager()
    num_workers = args.num_workers
    batch_size = args.batch_size
    output_queue = manager.Queue(maxsize=num_workers * 100)
    convert_parquet_to_wds(args)
    write_config(EPath(args.output_dir).absolute(), args.media)
