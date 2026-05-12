import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import re

matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang HK', 'simhei']
matplotlib.rcParams['axes.unicode_minus'] = False

def plot_multigame_averages(excel_file, output_dir="Plot_Results"):
    try:
        df = pd.read_excel(excel_file)
    except FileNotFoundError:
        print(f"找不到檔案: {excel_file}，請確認是否已產出批次追蹤資料")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    individual_dir = os.path.join(output_dir, "Individual_Features")
    os.makedirs(individual_dir, exist_ok=True)
    
    turn_col = 'feat_a_巡數' if 'feat_a_巡數' in df.columns else '當下巡數'

    turn_counts = df[turn_col].value_counts()
    valid_turns = turn_counts[turn_counts >= 5].index
    df_valid = df[df[turn_col].isin(valid_turns)]

    features_to_calc = [
        'feat_b_吃碰數',
        'feat_c_花色集中度',
        'feat_d_中張比例(3 ~ 7)',
        'feat_e_邊張比例(1、2、8、9)',
        'feat_f_字牌比例',
        'feat_g_摸切比例',
        'feat_h_連續摸切強度',
        'feat_i_摸切轉手切',
        'feat_j_中張第一張被打出',
        'feat_k_中張第二張被打出',
        'feat_l_中張第三張被打出',
        'feat_m_中張第四張被打出',
        'feat_n_字牌第一張被打出',
        'feat_o_字牌第二張被打出',
        'feat_p_字牌第三張被打出',
        'feat_q_字牌第四張被打出',
        'feat_r_邊張(1、9)第一張被打出',
        'feat_s_邊張(1、9)第二張被打出',
        'feat_t_邊張(1、9)第三張被打出',
        'feat_u_邊張(1、9)第四張被打出',
        'feat_v_邊張(2、8)第一張被打出',
        'feat_w_邊張(2、8)第二張被打出',
        'feat_x_邊張(2、8)第三張被打出',
        'feat_y_邊張(2、8)第四張被打出',
        '預測聽牌分數'
    ]

    features_to_calc = [f for f in features_to_calc if f in df_valid.columns]
    
    df_grouped = df_valid.groupby(turn_col)[features_to_calc].agg(['mean', 'std'])

    plt.figure(figsize=(18, 12))
    
    plot_styles = {
        '預測聽牌分數': {'color': 'black', 'marker': 'x', 'lw': 3.5, 'ls': '--'},
        'feat_b_吃碰數': {'color': 'tab:pink', 'marker': 'o', 'lw': 1.5, 'ls': '-'},
        'feat_c_花色集中度': {'color': 'tab:orange', 'marker': 's', 'lw': 2, 'ls': '-'},
        'feat_d_中張比例(3 ~ 7)': {'color': 'gold', 'marker': '^', 'lw': 2, 'ls': '-'},
        'feat_e_邊張比例(1、2、8、9)': {'color': 'tab:olive', 'marker': 'v', 'lw': 2, 'ls': '-'},
        'feat_f_字牌比例': {'color': 'yellowgreen', 'marker': '<', 'lw': 2, 'ls': '-'},
        'feat_g_摸切比例': {'color': 'tab:red', 'marker': 'D', 'lw': 2, 'ls': '-'},
        'feat_h_連續摸切強度': {'color': 'tab:purple', 'marker': 'p', 'lw': 2, 'ls': '-'},
        'feat_i_摸切轉手切': {'color': 'tab:brown', 'marker': '*', 'lw': 2, 'ls': '-'},
        
        'feat_j_中張第一張被打出': {'color': 'tab:blue', 'marker': '1', 'lw': 1.5, 'ls': '-'},
        'feat_k_中張第二張被打出': {'color': 'tab:blue', 'marker': '2', 'lw': 1.5, 'ls': '-'},
        'feat_l_中張第三張被打出': {'color': 'tab:blue', 'marker': '3', 'lw': 1.5, 'ls': '-'},
        'feat_m_中張第四張被打出': {'color': 'tab:blue', 'marker': '4', 'lw': 1.5, 'ls': '-'},
        
        'feat_n_字牌第一張被打出': {'color': 'tab:green', 'marker': '1', 'lw': 1.5, 'ls': '-'},
        'feat_o_字牌第二張被打出': {'color': 'tab:green', 'marker': '2', 'lw': 1.5, 'ls': '-'},
        'feat_p_字牌第三張被打出': {'color': 'tab:green', 'marker': '3', 'lw': 1.5, 'ls': '-'},
        'feat_q_字牌第四張被打出': {'color': 'tab:green', 'marker': '4', 'lw': 1.5, 'ls': '-'},
        
        'feat_r_邊張(1、9)第一張被打出': {'color': 'darkcyan', 'marker': '1', 'lw': 1.5, 'ls': '-'},
        'feat_s_邊張(1、9)第二張被打出': {'color': 'darkcyan', 'marker': '2', 'lw': 1.5, 'ls': '-'},
        'feat_t_邊張(1、9)第三張被打出': {'color': 'darkcyan', 'marker': '3', 'lw': 1.5, 'ls': '-'},
        'feat_u_邊張(1、9)第四張被打出': {'color': 'darkcyan', 'marker': '4', 'lw': 1.5, 'ls': '-'},
        
        'feat_v_邊張(2、8)第一張被打出': {'color': 'indigo', 'marker': '1', 'lw': 1.5, 'ls': '-'},
        'feat_w_邊張(2、8)第二張被打出': {'color': 'indigo', 'marker': '2', 'lw': 1.5, 'ls': '-'},
        'feat_x_邊張(2、8)第三張被打出': {'color': 'indigo', 'marker': '3', 'lw': 1.5, 'ls': '-'},
        'feat_y_邊張(2、8)第四張被打出': {'color': 'indigo', 'marker': '4', 'lw': 1.5, 'ls': '-'}
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
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, ncol=1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "All_Games_Average_Combined.png"), dpi=300)
    plt.close()

    print(f"綜合圖表已儲存。正在為 {len(features_to_calc)} 個特徵生成獨立研究圖表...")
    
    for feature in features_to_calc:
        if feature not in plot_styles:
            continue
            
        style = plot_styles[feature]
        x_vals = df_grouped.index
        y_mean = df_grouped[feature]['mean']
        y_std = df_grouped[feature]['std'].fillna(0)
        
        plt.figure(figsize=(12, 7))
        
        plt.plot(x_vals, y_mean, label='平均值 (Mean)', color=style['color'], 
                 marker=style['marker'], linewidth=3)
      
        plt.fill_between(x_vals, y_mean - y_std, y_mean + y_std, 
                         color=style['color'], alpha=0.15, label='±1 標準差 (Std Dev)')

        clean_title = feature.replace('feat_', '')
        plt.title(f'數據研究：{clean_title} 隨巡數之變化趨勢', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('當下巡數 (Turn)', fontsize=13)
        plt.ylabel('特徵數值 / 預測分數', fontsize=13)
        
        plt.xticks(x_vals, fontsize=10)
        plt.yticks(fontsize=10)
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='upper left', fontsize=11)
     
        safe_filename = feature.replace('預測聽牌分數', 'Score').replace('feat_', '')
        safe_filename = re.sub(r'[\\/:*?"<>|、()~ ]', '_', safe_filename).strip('_')
        
        save_path = os.path.join(individual_dir, f"Detail_Trend_{safe_filename}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"✅ 所有獨立圖檔已成功儲存至: {individual_dir}")


if __name__ == "__main__":
    EXCEL_FILE = "Batch_Linear_Tracking_Result.xlsx"
    OUTPUT_FOLDER = "Plot_Results"                   
    plot_multigame_averages(EXCEL_FILE, OUTPUT_FOLDER)