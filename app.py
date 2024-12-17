import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import pandas as pd
import matplotlib.pyplot as plt

# 공통 sample_id 추출 함수
def get_common_sample_ids(dataframes):
    """
    여러 데이터프레임에서 공통 sample_id 추출
    dataframes: 데이터프레임 딕셔너리
    """
    sample_id_sets = [set(df['sample_id']) for df in dataframes.values() if df is not None]
    return set.intersection(*sample_id_sets) if sample_id_sets else set()

# 그래프 생성 함수 (공통 sample_id 기준)
def create_and_plot_graphs_filtered(dataframes, selected_sample_id):
    """
    특정 sample_id를 기준으로 필터링된 데이터를 사용해 그래프 생성
    """
    figsize = (10, 8)
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
    
    # Sigma 그래프
    df_combined_sigma = dataframes.get('sigma')
    if df_combined_sigma is not None:
        df_filtered_sigma = df_combined_sigma[df_combined_sigma['sample_id'] == selected_sample_id]
        if not df_filtered_sigma.empty:
            df_filtered_sigma = df_filtered_sigma.sort_values(by='temperature')
            ax1.plot(df_filtered_sigma['temperature'], df_filtered_sigma['tepvalue'], marker='o', linestyle='-', color='m')
            ax1.set_title('Sigma')
            ax1.set_xlabel('Temperature (K)')
            ax1.set_ylabel(r'$\sigma$ $[S/cm]$')
            ax1.grid(True)

    # Alpha 그래프
    df_alpha = dataframes.get('alpha')
    if df_alpha is not None:
        df_filtered_alpha = df_alpha[df_alpha['sample_id'] == selected_sample_id]
        if not df_filtered_alpha.empty:
            df_filtered_alpha = df_filtered_alpha.sort_values(by='temperature')
            ax2.plot(df_filtered_alpha['temperature'], df_filtered_alpha['tepvalue'] * 1e6, marker='o', linestyle='-', color='g')
            ax2.set_title('Alpha')
            ax2.set_xlabel('Temperature (K)')
            ax2.set_ylabel(r'$\alpha$ $[\mu V/K]$')
            ax2.grid(True)

    # Kappa 그래프
    df_kappa = dataframes.get('kappa')
    if df_kappa is not None:
        df_filtered_k = df_kappa[df_kappa['sample_id'] == selected_sample_id]
        if not df_filtered_k.empty:
            df_filtered_k = df_filtered_k.sort_values(by='temperature')
            ax3.plot(df_filtered_k['temperature'], df_filtered_k['tepvalue'], marker='o', linestyle='-', color='r')
            ax3.set_title('Kappa')
            ax3.set_xlabel('Temperature (K)')
            ax3.set_ylabel(r'$k$ $[W/(m·K)]$')
            ax3.grid(True)

    # ZT 그래프
    df_combined_ZT = dataframes.get('ZT')
    if df_combined_ZT is not None:
        df_filtered_ZT = df_combined_ZT[df_combined_ZT['sample_id'] == selected_sample_id]
        if not df_filtered_ZT.empty:
            df_filtered_ZT = df_filtered_ZT.sort_values(by='temperature')
            ax4.plot(df_filtered_ZT['temperature'], df_filtered_ZT['tepvalue'], marker='o', linestyle='-', color='b')
            ax4.set_title('ZT')
            ax4.set_xlabel('Temperature (K)')
            ax4.set_ylabel(r'$ZT$')
            ax4.grid(True)

    plt.tight_layout()
    plt.show()

# 데이터프레임 불러오기
df_combined_sigma = pd.read_csv('df_combined_sigma.csv')
df_alpha = pd.read_csv('df_alpha0.csv')
df_kappa = pd.read_csv('df_kappa0.csv')
df_combined_ZT = pd.read_csv('df_combined_ZT.csv')

# 데이터프레임 딕셔너리 구성
dataframes = {
    'sigma': df_combined_sigma,
    'alpha': df_alpha,
    'kappa': df_kappa,
    'ZT': df_combined_ZT
}

# 공통 sample_id 추출 및 출력
common_sample_ids = get_common_sample_ids(dataframes)
print("Common Sample IDs:", common_sample_ids)

# 특정 sample_id 그래프 출력
if common_sample_ids:
    # 첫 번째 공통 sample_id를 선택해서 그래프 생성
    selected_sample_id = list(common_sample_ids)[0]
    print(f"\nSelected Sample ID for Plotting: {selected_sample_id}")
    create_and_plot_graphs_filtered(dataframes, selected_sample_id)
else:
    print("No common sample IDs found.")
