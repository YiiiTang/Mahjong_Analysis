import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang HK', 'simhei']
matplotlib.rcParams['axes.unicode_minus'] = False

def plot_multigame_averages(excel_file, output_dir="Plot_Results"):
    try:
        df = pd.read_excel(excel_file)
    except FileNotFoundError:
        print(f"找不到檔案: {excel_file}，請確認是否已產出批次追蹤資料")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    turn_counts = df['當下巡數'].value_counts()
    valid_turns = turn_counts[turn_counts >= 5].index
    df_valid = df[df['當下巡數'].isin(valid_turns)]

    # [新增] 將 4 個新特徵加入計算清單
    features_to_calc = [
        '中張比例', '花色集中度', '字牌比例', '摸切比例', 
        '連續摸切強度', '摸切轉手切', '第一張被打出', '第二張被打出', 
        '第三張被打出', '第四張被打出', '預測聽牌分數'
    ]
    
    df_grouped = df_valid.groupby('當下巡數')[features_to_calc].agg(['mean', 'std'])

    plt.figure(figsize=(15, 9))
    
    # [新增] 為新特徵配置顏色與樣式
    plot_styles = {
        '預測聽牌分數': {'color': 'black', 'marker': 'x', 'lw': 3.5, 'ls': '--'},
        '中張比例': {'color': 'tab:blue', 'marker': 'o', 'lw': 2, 'ls': '-'},
        '花色集中度': {'color': 'tab:orange', 'marker': 's', 'lw': 2, 'ls': '-'},
        '字牌比例': {'color': 'tab:green', 'marker': '^', 'lw': 2, 'ls': '-'},
        '摸切比例': {'color': 'tab:red', 'marker': 'D', 'lw': 2, 'ls': '-'},
        '連續摸切強度': {'color': 'tab:purple', 'marker': 'v', 'lw': 2, 'ls': '-'},
        '摸切轉手切': {'color': 'tab:brown', 'marker': 'p', 'lw': 2, 'ls': '-'},
        '第一張被打出': {'color': 'tab:cyan', 'marker': '1', 'lw': 2, 'ls': '-'},
        '第二張被打出': {'color': 'tab:pink', 'marker': '2', 'lw': 2, 'ls': '-'},
        '第三張被打出': {'color': 'tab:olive', 'marker': '3', 'lw': 2, 'ls': '-'},
        '第四張被打出': {'color': 'darkcyan', 'marker': '4', 'lw': 2, 'ls': '-'}
    }

    for feature, style in plot_styles.items():
        if feature in df_grouped.columns.levels[0]:
            x_vals = df_grouped.index
            y_mean = df_grouped[feature]['mean']
            label = f'{feature} (平均)'
            
            plt.plot(x_vals, y_mean, label=label, 
                     color=style['color'], marker=style['marker'], 
                     linewidth=style['lw'], linestyle=style['ls'], alpha=0.85)

    plt.title('所有牌譜平均：特徵與聽牌預測之趨勢比較 (綜合圖)', fontsize=18, fontweight='bold', pad=15)
    plt.xlabel('當下巡數 (Turn)', fontsize=14)
    plt.ylabel('平均特徵值 / 分數', fontsize=14)
    plt.xticks(df_grouped.index)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12)
    plt.tight_layout()
    
    save_path_all = os.path.join(output_dir, "All_Games_Average_Combined.png")
    plt.savefig(save_path_all, dpi=300)
    print(f"✅ 已儲存綜合比較圖: {save_path_all}")
    plt.close()

    # [調整] 將子圖表改為 4x3 的排列方式，以容納 11 個指標
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 16), sharex=True)
    fig.suptitle('所有牌譜平均：各特徵變化趨勢與變異程度 (含標準差陰影)', fontsize=18, fontweight='bold')
    axes = axes.flatten()

    features_list = list(plot_styles.keys())
    
    for i, feature in enumerate(features_list):
        ax = axes[i]
        style = plot_styles[feature]
        x_vals = df_grouped.index
        y_mean = df_grouped[feature]['mean']
        y_std = df_grouped[feature]['std'].fillna(0)
        
        ax.plot(x_vals, y_mean, label='Mean', color=style['color'], 
                marker=style['marker'], linewidth=2.5)
        
        ax.fill_between(x_vals, y_mean - y_std, y_mean + y_std, 
                        color=style['color'], alpha=0.15, label='±1 Std Dev')

        ax.set_title(feature, fontsize=14)
        ax.set_ylabel('數值')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xticks(x_vals)
        ax.legend(loc='upper left', fontsize=10)

    # 隱藏多出來的第 12 個空白子圖表
    if len(axes) > len(features_list):
        for j in range(len(features_list), len(axes)):
            axes[j].set_visible(False)
    
    # 將最底層的 X 軸加上標籤
    axes[8].set_xlabel('當下巡數', fontsize=12)
    axes[9].set_xlabel('當下巡數', fontsize=12)
    axes[10].set_xlabel('當下巡數', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path_subplots = os.path.join(output_dir, "All_Games_Average_Subplots.png")
    plt.savefig(save_path_subplots, dpi=300)
    plt.close()


if __name__ == "__main__":
    EXCEL_FILE = "Batch_Linear_Tracking_Result.xlsx"
    OUTPUT_FOLDER = "Plot_Results"                   

    plot_multigame_averages(EXCEL_FILE, OUTPUT_FOLDER)