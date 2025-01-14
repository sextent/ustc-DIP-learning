import os

def append_dataset_lists(dataset_path, train_list='train_list.txt', val_list='val_list.txt'):
    """
    将新数据集的图片路径追加到现有的训练和验证列表中
    
    Args:
        dataset_path (str): 新数据集的根目录路径
        train_list (str): 训练列表文件名
        val_list (str): 验证列表文件名
    """
    # 确保数据集目录存在
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    
    # 追加训练集图片
    train_dir = os.path.join(dataset_path, 'train')
    if os.path.exists(train_dir):
        # 首先读取现有的训练列表
        existing_files = set()
        if os.path.exists(train_list):
            with open(train_list, 'r', encoding='utf-16') as f:
                existing_files = set(line.strip() for line in f)
        
        # 追加新的训练图片
        with open(train_list, 'a', encoding='utf-16') as f:
            for filename in sorted(os.listdir(train_dir)):
                if filename.endswith('.jpg'):
                    full_path = os.path.abspath(os.path.join(train_dir, filename))
                    # 只有当文件路径不在现有列表中时才添加
                    if full_path not in existing_files:
                        f.write(full_path + '\n')
                        print(f"Added to training list: {full_path}")
    
    # 追加验证集图片
    val_dir = os.path.join(dataset_path, 'val')
    if os.path.exists(val_dir):
        # 首先读取现有的验证列表
        existing_files = set()
        if os.path.exists(val_list):
            with open(val_list, 'r', encoding='utf-16') as f:
                existing_files = set(line.strip() for line in f)
        
        # 追加新的验证图片
        with open(val_list, 'a', encoding='utf-16') as f:
            for filename in sorted(os.listdir(val_dir)):
                if filename.endswith('.jpg'):
                    full_path = os.path.abspath(os.path.join(val_dir, filename))
                    # 只有当文件路径不在现有列表中时才添加
                    if full_path not in existing_files:
                        f.write(full_path + '\n')
                        print(f"Added to validation list: {full_path}")

def print_dataset_statistics(train_list='train_list.txt', val_list='val_list.txt'):
    """
    打印数据集统计信息
    """
    # 打印训练集统计
    if os.path.exists(train_list):
        with open(train_list, 'r', encoding='utf-16') as f:
            train_files = [line.strip() for line in f]
        print(f"\nTraining set statistics:")
        print(f"Total images: {len(train_files)}")
        print("Sample paths:")
        for path in train_files[:3]:  # 打印前3个样本路径
            print(f"  {path}")
    
    # 打印验证集统计
    if os.path.exists(val_list):
        with open(val_list, 'r', encoding='utf-16') as f:
            val_files = [line.strip() for line in f]
        print(f"\nValidation set statistics:")
        print(f"Total images: {len(val_files)}")
        print("Sample paths:")
        for path in val_files[:3]:  # 打印前3个样本路径
            print(f"  {path}")

if __name__ == '__main__':
    
    new_dataset_path = './datasets/cityscapes'  # 替换为新数据集的路径
    
    try:
        # 追加新数据集
        append_dataset_lists(new_dataset_path)
        
        # 打印更新后的统计信息
        print("\n=== Dataset Statistics After Update ===")
        print_dataset_statistics()
        
    except Exception as e:
        print(f"Error: {str(e)}")