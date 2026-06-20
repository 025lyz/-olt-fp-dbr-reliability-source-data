import numpy as np
from tmm import coh_tmm
import multiprocessing as mp
from tqdm import tqdm
import os
import time

# ==========================================
# 1. 物理参数与边界设定 (根据你的计划书)
# ==========================================
WAVELENGTHS = np.linspace(3000, 5000, 1000)  # 3-5 μm，离散化为200个波长点
NUM_SAMPLES = 1000  # 测试阶段设为 1000，台式机全量跑可改回 100000

# 结构参数搜索空间 (单位: nm)
BOUNDS = {
    'd_H': (100, 1000),
    'd_L': (100, 1500),
    'N': (2, 10),
    'L_c': (500, 5000)  # 0.5-5 μm 转换为 nm
}

# 假设的材料折射率 (中红外常用材料，可自行替换为你实际使用的材料)
N_AIR = 1.0
N_H = 3.42  # 例如: Si (硅)
N_L = 1.41  # 例如: SiO2 (二氧化硅)
N_CAVITY = 1.41  # 假设谐振腔也是低折射率材料
N_SUB = 3.42  # 衬底折射率


# ==========================================
# 2. 核心计算函数 (Worker)
# ==========================================
def simulate_single_structure(seed):
    """
    单个结构的生成与 TMM 仿真函数
    """
    # 重新初始化随机种子，防止多进程随机数冲突
    np.random.seed(seed)

    # 1. 在设定空间内随机采样
    d_H = np.random.uniform(*BOUNDS['d_H'])
    d_L = np.random.uniform(*BOUNDS['d_L'])
    N = int(np.random.randint(*BOUNDS['N']))
    L_c = np.random.uniform(*BOUNDS['L_c'])

    # 【可选】数据增强：加入极小的工艺误差噪声 (Thickness Noise)
    # noise_H = np.random.normal(0, 2) # +-2nm
    # d_H += noise_H ...

    # 2. 构建厚度和折射率数组 (Air / DBR1 / Cavity / DBR2 / Substrate)
    # DBR 周期通常为 H, L 交替
    dbr_thicknesses = [d_H, d_L] * N
    dbr_indices = [N_H, N_L] * N

    # 完整结构：Air + DBR1 + Cavity + DBR2 + Substrate
    # 注意：TMM 库要求第一层和最后一层（环境与衬底）厚度设为无穷大 (inf)
    thicknesses = [np.inf] + dbr_thicknesses + [L_c] + dbr_thicknesses + [np.inf]
    indices = [N_AIR] + dbr_indices + [N_CAVITY] + dbr_indices + [N_SUB]

    # 3. 运行 TMM 循环计算光谱
    transmission_spectrum = np.zeros(len(WAVELENGTHS))

    for i, wl in enumerate(WAVELENGTHS):
        # coh_tmm 参数: 偏振(s/p), 膜层折射率, 膜层厚度, 入射角, 波长
        # 假设为垂直入射 (0度)，所以偏振 s 或 p 结果一样
        res = coh_tmm('s', indices, thicknesses, 0.0, wl)
        transmission_spectrum[i] = res['T']  # 提取透射率

    params = np.array([d_H, d_L, N, L_c])
    return params, transmission_spectrum


# ==========================================
# 3. 多进程调度与保存
# ==========================================
def main():
    print(f"开始生成数据集，总样本数: {NUM_SAMPLES}")
    print(f"目标波段: {WAVELENGTHS[0]} - {WAVELENGTHS[-1]} nm, 采样点: {len(WAVELENGTHS)}")

    start_time = time.time()

    # 自动获取当前机器的 CPU 核心数，留一个核心防止电脑卡死
    num_cores = max(1, mp.cpu_count() - 1)
    print(f"启用多进程加速，使用核心数: {num_cores}")

    # 准备随机种子池
    seeds = np.random.randint(0, 2 ** 31 - 1, size=NUM_SAMPLES)

    # 初始化存储容器
    all_params = np.zeros((NUM_SAMPLES, 4))  # [d_H, d_L, N, L_c]
    all_spectra = np.zeros((NUM_SAMPLES, len(WAVELENGTHS)))

    # 开启进程池，并使用 tqdm 显示进度条
    with mp.Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap(simulate_single_structure, seeds), total=NUM_SAMPLES))

    # 解包结果
    for i, (params, spec) in enumerate(results):
        all_params[i] = params
        all_spectra[i] = spec

    # 创建保存目录
    os.makedirs('dataset', exist_ok=True)
    save_path = f'dataset/fp_dbr_data_{NUM_SAMPLES}.npz'

    # 将数据压缩保存，方便后续 PyTorch 读取
    np.savez_compressed(
        save_path,
        params=all_params,
        spectra=all_spectra,
        wavelengths=WAVELENGTHS
    )

    elapsed = time.time() - start_time
    print(f"数据集生成完毕！保存在: {save_path}")
    print(f"总耗时: {elapsed:.2f} 秒 (平均 {elapsed / NUM_SAMPLES * 1000:.2f} ms / 样本)")


if __name__ == '__main__':
    main()