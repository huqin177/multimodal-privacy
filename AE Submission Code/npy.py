# -*- coding: utf-8 -*-
"""
NPY文件内容详细查看器
可以查看特征包中的所有数据
"""
import numpy as np
import os
import json
from pprint import pprint

def view_npy_content(file_path):
    """详细查看NPY文件内容"""
    
    print("="*70)
    print(f"📁 文件: {file_path}")
    print("="*70)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return None
    
    # 获取文件大小
    file_size = os.path.getsize(file_path)
    print(f"📊 文件大小: {file_size:,} 字节 ({file_size/1024:.2f} KB)")
    
    # 加载数据
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"✅ 加载成功")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None
    
    # 数据类型和结构
    print(f"\n📦 数据类型: {type(data)}")
    print(f"📏 数据形状: {data.shape if hasattr(data, 'shape') else 'N/A'}")
    print(f"🔢 数据数量: {len(data) if hasattr(data, '__len__') else 'N/A'}")
    
    # 如果是列表或数组，遍历每个元素
    if hasattr(data, '__len__') and len(data) > 0:
        print(f"\n{'='*70}")
        print(f"📋 共有 {len(data)} 个实体")
        print(f"{'='*70}")
        
        for idx, item in enumerate(data):
            print(f"\n{'─'*70}")
            print(f"🔸 实体 {idx + 1}/{len(data)}")
            print(f"{'─'*70}")
            
            # 判断item类型
            if isinstance(item, dict):
                # 打印所有键
                print(f"📌 字段列表: {list(item.keys())}")
                print()
                
                # 详细打印每个字段
                for key, value in item.items():
                    print(f"  【{key}】")
                    
                    # 根据值的类型打印不同格式
                    if isinstance(value, dict):
                        print(f"    类型: dict")
                        print(f"    内容:")
                        for sub_key, sub_val in value.items():
                            if isinstance(sub_val, np.ndarray):
                                print(f"      {sub_key}: shape={sub_val.shape}, dtype={sub_val.dtype}")
                                if sub_val.size < 100:  # 小数组显示数值
                                    print(f"        值: {sub_val}")
                            elif isinstance(sub_val, (list, tuple)) and len(sub_val) > 10:
                                print(f"      {sub_key}: {sub_val[:5]}... (共{len(sub_val)}个)")
                            else:
                                print(f"      {sub_key}: {sub_val}")
                    
                    elif isinstance(value, np.ndarray):
                        print(f"    类型: numpy数组")
                        print(f"    形状: {value.shape}")
                        print(f"    数据类型: {value.dtype}")
                        print(f"    维度: {value.ndim}D")
                        if value.size < 100:  # 小数组显示具体数值
                            print(f"    数值: {value}")
                        elif value.ndim == 1 and value.size <= 20:
                            print(f"    数值: {value}")
                        else:
                            # 显示统计信息
                            print(f"    最小值: {value.min():.4f}")
                            print(f"    最大值: {value.max():.4f}")
                            print(f"    平均值: {value.mean():.4f}")
                            print(f"    标准差: {value.std():.4f}")
                            # 显示前10个值
                            print(f"    前10个值: {value.flatten()[:10]}")
                    
                    elif isinstance(value, (list, tuple)):
                        print(f"    类型: {type(value).__name__}")
                        print(f"    长度: {len(value)}")
                        if len(value) <= 20:
                            print(f"    内容: {value}")
                        else:
                            print(f"    前5个: {value[:5]}")
                            print(f"    后5个: {value[-5:]}")
                    
                    elif isinstance(value, str):
                        print(f"    类型: 字符串")
                        print(f"    长度: {len(value)}")
                        if len(value) <= 200:
                            print(f"    内容: {value}")
                        else:
                            print(f"    内容: {value[:200]}...")
                    
                    elif value is None:
                        print(f"    类型: None")
                    
                    else:
                        print(f"    类型: {type(value).__name__}")
                        print(f"    值: {value}")
                    
                    print()
            
            elif isinstance(item, (list, tuple)):
                print(f"📌 这是一个列表/元组，长度: {len(item)}")
                if len(item) <= 10:
                    for i, sub_item in enumerate(item):
                        print(f"  [{i}]: {type(sub_item).__name__}")
                        if isinstance(sub_item, dict):
                            for k, v in sub_item.items():
                                if isinstance(v, np.ndarray):
                                    print(f"      {k}: shape={v.shape}")
                                else:
                                    print(f"      {k}: {v}")
                        else:
                            print(f"      {sub_item}")
                else:
                    print(f"  前3个元素:")
                    for i in range(min(3, len(item))):
                        print(f"    [{i}]: {type(item[i]).__name__}")
            
            elif isinstance(item, np.ndarray):
                print(f"📌 numpy数组")
                print(f"  形状: {item.shape}")
                print(f"  数据类型: {item.dtype}")
                if item.size <= 50:
                    print(f"  数值: {item}")
                else:
                    print(f"  前10个值: {item.flatten()[:10]}")
            
            else:
                print(f"📌 值: {item}")
    
    else:
        # 单个数据项
        print(f"\n📋 数据内容:")
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"  {key}: {type(value).__name__}")
                if isinstance(value, np.ndarray):
                    print(f"    shape: {value.shape}")
        else:
            print(f"  {data}")
    
    return data


def save_as_json(file_path, output_path=None):
    """将NPY文件保存为JSON格式，方便查看"""
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return
    
    # 加载数据
    data = np.load(file_path, allow_pickle=True)
    
    # 转换为可JSON序列化的格式
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    # 转换数据
    if isinstance(data, (list, np.ndarray)):
        serializable_data = [convert_to_serializable(item) for item in data]
    else:
        serializable_data = convert_to_serializable(data)
    
    # 设置输出路径
    if output_path is None:
        output_path = file_path.replace('.npy', '.json')
    
    # 保存为JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ JSON已保存: {output_path}")
    return output_path


def batch_view_directory(directory):
    """批量查看目录下所有NPY文件"""
    
    npy_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):
                npy_files.append(os.path.join(root, file))
    
    if not npy_files:
        print(f"❌ 在 {directory} 中未找到任何 .npy 文件")
        return
    
    print(f"\n{'='*70}")
    print(f"🔍 找到 {len(npy_files)} 个 NPY 文件")
    print(f"{'='*70}")
    
    for i, npy_file in enumerate(npy_files, 1):
        print(f"\n{'='*70}")
        print(f"📄 [{i}/{len(npy_files)}] {npy_file}")
        print(f"{'='*70}")
        
        # 查看内容
        data = view_npy_content(npy_file)
        
        # 询问是否导出JSON
        if data is not None and len(data) > 0:
            response = input(f"\n是否导出为JSON? (y/n): ").strip().lower()
            if response == 'y':
                save_as_json(npy_file)
        
        print("\n" + "─"*70)
        input("按 Enter 继续查看下一个文件...")


# ==================== 主程序 ====================
if __name__ == "__main__":
    import sys
    
    print("="*70)
    print("🔍 NPY文件内容查看器")
    print("="*70)
    print()
    
    # 使用方式
    if len(sys.argv) > 1:
        # 命令行指定文件或目录
        path = sys.argv[1]
        
        if os.path.isdir(path):
            # 如果是目录，批量查看
            batch_view_directory(path)
        elif os.path.isfile(path) and path.endswith('.npy'):
            # 如果是文件，查看单个
            view_npy_content(path)
            
            # 询问是否导出JSON
            response = input("\n是否导出为JSON? (y/n): ").strip().lower()
            if response == 'y':
                save_as_json(path)
        else:
            print(f"❌ 无效路径: {path}")
    else:
        # 交互式选择
        print("请选择查看方式:")
        print("1. 查看当前目录下所有NPY文件")
        print("2. 手动输入文件路径")
        print("3. 自动搜索整个项目目录")
        
        choice = input("\n请输入选项 (1/2/3): ").strip()
        
        if choice == '1':
            # 当前目录
            current_dir = os.getcwd()
            print(f"\n搜索目录: {current_dir}")
            batch_view_directory(current_dir)
        
        elif choice == '2':
            # 手动输入路径
            file_path = input("请输入NPY文件路径: ").strip()
            if os.path.exists(file_path):
                view_npy_content(file_path)
                response = input("\n是否导出为JSON? (y/n): ").strip().lower()
                if response == 'y':
                    save_as_json(file_path)
            else:
                print(f"❌ 文件不存在: {file_path}")
        
        elif choice == '3':
            # 自动搜索项目目录
            search_path = r"D:\办公\word\大创\project\AE Submission Code"
            print(f"\n搜索目录: {search_path}")
            batch_view_directory(search_path)
        
        else:
            print("❌ 无效选项")