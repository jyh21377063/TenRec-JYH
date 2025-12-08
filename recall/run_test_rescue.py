import torch
import os
import pickle
from torch.utils.data import DataLoader
from dataset import SBRDataset
from model.dssm.dssm import TwoTowerModel
from evaluation import Evaluator

# ==========================
# 配置区域 (对应你报错的路径)
# ==========================
CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 4096,
    'best_model_path': r'../../experiments/EXP_20251208_151742/checkpoints/best_model.pth',
    'data_path': '../../data/sbr_data_1208.pkl',
}


def rescue_test():
    device = torch.device(CONFIG['device'])
    print(f"🚀 开始抢救性测试，使用设备: {device}")

    # 1. 检查模型文件是否存在
    if not os.path.exists(CONFIG['best_model_path']):
        print(f"❌ 错误：找不到模型文件: {CONFIG['best_model_path']}")
        print("请手动修改脚本中的 'best_model_path' 为正确的 .pth 文件路径")
        return

    # 2. 加载数据 (为了获取 meta 信息和测试集)
    print("📥 正在加载数据...")
    with open(CONFIG['data_path'], 'rb') as f:
        all_data = pickle.load(f)

    # 只需要测试集
    test_ds = SBRDataset(data=all_data, mode='test')
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)  # Windows下设为0更安全

    # 获取 Meta 信息用于初始化模型
    # 注意：这里我们重新从数据获取 meta，保证和训练时一致
    train_ds_temp = SBRDataset(data=all_data, mode='train')
    meta = train_ds_temp.get_meta()

    # 构建 Full Dataset 字典供 Evaluator 使用
    def extract_ids(mode_data):
        return {'user_id': mode_data['user_id'], 'item_id': mode_data['item_id']}

    full_dataset = {
        'train': extract_ids(all_data['train']),
        'test': extract_ids(all_data['test']),
        'val': extract_ids(all_data['val']),
        'meta': meta
    }

    # 清理内存
    del all_data, train_ds_temp

    # 3. 初始化模型
    print("🛠️  初始化模型...")
    model = TwoTowerModel(meta).to(device)

    # 4. 加载权重 (关键步骤：使用了 weights_only=False)
    print(f"💾 加载模型权重: {CONFIG['best_model_path']}")
    try:
        # !!! 这里的 weights_only=False 就是修复的核心 !!!
        checkpoint = torch.load(CONFIG['best_model_path'], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 5. 开始评估
    print("📉 开始最终测试评估...")
    evaluator_test = Evaluator(model, full_dataset, device, k_list=[10, 20, 50])

    model.eval()
    test_results = evaluator_test.evaluate(test_loader)

    # 6. 输出结果
    print("\n" + "=" * 50)
    print("🎉 最终测试结果 (FINAL TEST RESULTS)")
    print("=" * 50)
    for k, v in test_results.items():
        print(f"{k}: {v:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    rescue_test()