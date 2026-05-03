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
    
    turn_col = 'feat_a_巡數' if 'feat_a_巡數' in df.columns else '當下巡數'

    turn_counts = df[turn_col].value_counts()
    valid_turns = turn_counts[turn_counts >= 5].index
    df_valid = df[df[turn_col].isin(valid_turns)]

    features_to_calc = [
        'feat_b1_一副露', 'feat_b2_二副露', 'feat_b3_三副露', 'feat_b4_四副露',
        'feat_c_花色集中度', 'feat_d_摸切比例', 'feat_e_連續摸切強度', 'feat_f_摸切轉手切',
        'feat_g_中張第一張被打出', 'feat_h_中張第二張被打出', 'feat_i_中張第三張被打出', 'feat_j_中張第四張被打出',
        'feat_k_字牌第一張被打出', 'feat_l_字牌第二張被打出', 'feat_m_字牌第三張被打出', 'feat_n_字牌第四張被打出',
        'feat_o_邊張第一張被打出', 'feat_p_邊張第二張被打出', 'feat_q_邊張第三張被打出', 'feat_r_邊張第四張被打出',
        '預測聽牌分數'
    ]

    features_to_calc = [f for f in features_to_calc if f in df_valid.columns]
    
    df_grouped = df_valid.groupby(turn_col)[features_to_calc].agg(['mean', 'std'])

    plt.figure(figsize=(16, 10))
    
    plot_styles = {
        '預測聽牌分數': {'color': 'black', 'marker': 'x', 'lw': 3.5, 'ls': '--'},
        'feat_b1_一副露': {'color': 'tab:pink', 'marker': '*', 'lw': 1.5, 'ls': '-'},
        'feat_b2_二副露': {'color': 'hotpink', 'marker': '*', 'lw': 1.5, 'ls': '-'},
        'feat_b3_三副露': {'color': 'deeppink', 'marker': '*', 'lw': 1.5, 'ls': '-'},
        'feat_b4_四副露': {'color': 'crimson', 'marker': '*', 'lw': 1.5, 'ls': '-'},
        'feat_c_花色集中度': {'color': 'tab:orange', 'marker': 's', 'lw': 2, 'ls': '-'},
        'feat_d_摸切比例': {'color': 'tab:red', 'marker': 'D', 'lw': 2, 'ls': '-'},
        'feat_e_連續摸切強度': {'color': 'tab:purple', 'marker': 'v', 'lw': 2, 'ls': '-'},
        'feat_f_摸切轉手切': {'color': 'tab:brown', 'marker': 'p', 'lw': 2, 'ls': '-'},
        'feat_g_中張第一張被打出': {'color': 'tab:blue', 'marker': '1', 'lw': 1.5, 'ls': '-'},
        'feat_h_中張第二張被打出': {'color': 'tab:blue', 'marker': '2', 'lw': 1.5, 'ls': '-'},
        'feat_i_中張第三張被打出': {'color': 'tab:blue', 'marker': '3', 'lw': 1.5, 'ls': '-'},
        'feat_j_中張第四張被打出': {'color': 'tab:blue', 'marker': '4', 'lw': 1.5, 'ls': '-'},
        'feat_k_字牌第一張被打出': {'color': 'tab:green', 'marker': '1', 'lw': 1.5, 'ls': '-'},
        'feat_l_字牌第二張被打出': {'color': 'tab:green', 'marker': '2', 'lw': 1.5, 'ls': '-'},
        'feat_m_字牌第三張被打出': {'color': 'tab:green', 'marker': '3', 'lw': 1.5, 'ls': '-'},
        'feat_n_字牌第四張被打出': {'color': 'tab:green', 'marker': '4', 'lw': 1.5, 'ls': '-'},
        'feat_o_邊張第一張被打出': {'color': 'darkcyan', 'marker': '1', 'lw': 1.5, 'ls': '-'},
        'feat_p_邊張第二張被打出': {'color': 'darkcyan', 'marker': '2', 'lw': 1.5, 'ls': '-'},
        'feat_q_邊張第三張被打出': {'color': 'darkcyan', 'marker': '3', 'lw': 1.5, 'ls': '-'},
        'feat_r_邊張第四張被打出': {'color': 'darkcyan', 'marker': '4', 'lw': 1.5, 'ls': '-'}
    }

    for feature, style in plot_styles.items():
        if feature in df_grouped.columns.levels[0]:
            x_vals = df_grouped.index
            y_mean = df_grouped[feature]['mean']
            label = f'{feature.replace("feat_", "")}'
            
            plt.plot(x_vals, y_mean, label=label, 
                     color=style['color'], marker=style['marker'], 
                     linewidth=style['lw'], linestyle=style['ls'], alpha=0.85)

    plt.title('所有牌譜平均：特徵與聽牌預測之趨勢比較 (綜合圖)', fontsize=18, fontweight='bold', pad=15)
    plt.xlabel('當下巡數 (Turn)', fontsize=14)
    plt.ylabel('平均特徵值 / 分數', fontsize=14)
    plt.xticks(df_grouped.index)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
    plt.tight_layout()
    
    save_path_all = os.path.join(output_dir, "All_Games_Average_Combined.png")
    plt.savefig(save_path_all, dpi=300)
    print(f"✅ 已儲存綜合比較圖: {save_path_all}")
    plt.close()

    fig, axes = plt.subplots(nrows=7, ncols=3, figsize=(18, 25), sharex=True)
    fig.suptitle('所有牌譜平均：各特徵變化趨勢與變異程度 (含標準差陰影)', fontsize=18, fontweight='bold')
    axes = axes.flatten()

    features_list = [f for f in list(plot_styles.keys()) if f in df_grouped.columns.levels[0]]
    
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

        ax.set_title(feature.replace('feat_', ''), fontsize=12)
        ax.set_ylabel('數值')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xticks(x_vals)
        ax.legend(loc='upper left', fontsize=9)

    if len(axes) > len(features_list):
        for j in range(len(features_list), len(axes)):
            axes[j].set_visible(False)

    for j in range(len(axes) - 3, len(axes)):
        axes[j].set_xlabel('當下巡數', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path_subplots = os.path.join(output_dir, "All_Games_Average_Subplots.png")
    plt.savefig(save_path_subplots, dpi=300)
    plt.close()

if __name__ == "__main__":
    EXCEL_FILE = "Batch_Linear_Tracking_Result.xlsx"
    OUTPUT_FOLDER = "Plot_Results"                   

    plot_multigame_averages(EXCEL_FILE, OUTPUT_FOLDER)