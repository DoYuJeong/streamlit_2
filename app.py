import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ZT 데이터 병합 및 그래프 생성 함수
def process_zt_data(df_ZT, df_Z, df_PF):
    all_zt_data = []

    # ZT 데이터 그대로 사용
    if df_ZT is not None:
        all_zt_data.append(df_ZT)

    # Z 데이터를 ZT로 변환
    if df_Z is not None:
        df_Z['tepvalue'] = df_Z['tepvalue'] * df_Z['temperature']
        all_zt_data.append(df_Z)

    # PF 데이터를 ZT로 변환
    if df_PF is not None:
        df_PF['tepvalue'] = df_PF['tepvalue'] / df_PF['temperature']
        all_zt_data.append(df_PF)

    # 병합 후 반환
    if all_zt_data:
        return pd.concat(all_zt_data, ignore_index=True).sort_values(by='temperature')
    return None

# Sigma 데이터 병합 및 그래프 생성 함수
def process_sigma_data(df_sigma, df_rho):
    if df_rho is not None:
        df_rho['tepvalue'] = 1 / df_rho['tepvalue']  # 역수로 변환
        df_rho['tepname'] = 'sigma'
        return pd.concat([df_sigma, df_rho], ignore_index=True).sort_values(by='temperature')
    return df_sigma

# 공통 sample_id 필터링 함수
def filter_common_samples_for_properties(dataframes, keys):
    """
    dataframes: 데이터프레임 딕셔너리
    keys: 공통 sample_id를 확인할 데이터프레임 키 리스트
    """
    filtered_dataframes = {key: dataframes[key] for key in keys if key in dataframes and dataframes[key] is not None}
    sample_sets = [set(df['sample_id']) for df in filtered_dataframes.values()]
    if sample_sets:
        return set.intersection(*sample_sets)
    return set()

# 그래프 생성 함수 수정
def create_and_plot_graphs(dataframes, selected_sample_id):
    figsize = (10, 8)
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    # Sigma 데이터 처리
    sigma_keys = ['sigma', 'rho']
    common_sigma_samples = filter_common_samples_for_properties(dataframes, sigma_keys)
    if selected_sample_id in common_sigma_samples:
        df_sigma = process_sigma_data(dataframes.get('sigma'), dataframes.get('rho'))
        df_sigma = df_sigma[df_sigma['sample_id'] == selected_sample_id]
        if not df_sigma.empty:
            ax1.plot(df_sigma['temperature'], df_sigma['tepvalue'], marker='o', linestyle='-', color='m')
            ax1.set_title(r'$\sigma$: Electrical Conductivity', fontsize=10)
            ax1.set_xlabel('Temperature (K)', fontsize=9)
            ax1.set_ylabel(r'$\sigma$ $[S/cm]$', fontsize=9)
            ax1.grid(True)

    # Alpha 데이터 처리
    alpha_keys = ['alpha']
    common_alpha_samples = filter_common_samples_for_properties(dataframes, alpha_keys)
    if selected_sample_id in common_alpha_samples:
        df_alpha = dataframes.get('alpha')
        df_alpha = df_alpha[df_alpha['sample_id'] == selected_sample_id]
        if not df_alpha.empty:
            ax2.plot(df_alpha['temperature'], df_alpha['tepvalue'] * 1e6, marker='o', linestyle='-', color='g')
            ax2.set_title(r'$\alpha$: Seebeck Coefficient', fontsize=10)
            ax2.set_xlabel('Temperature (K)', fontsize=9)
            ax2.set_ylabel(r'$\alpha$ $[\mu V/K]$', fontsize=9)
            ax2.grid(True)

    # Kappa 데이터 처리
    kappa_keys = ['kappa']
    common_kappa_samples = filter_common_samples_for_properties(dataframes, kappa_keys)
    if selected_sample_id in common_kappa_samples:
        df_kappa = dataframes.get('kappa')
        df_kappa = df_kappa[df_kappa['sample_id'] == selected_sample_id]
        if not df_kappa.empty:
            ax3.plot(df_kappa['temperature'], df_kappa['tepvalue'], marker='o', linestyle='-', color='r')
            ax3.set_title(r'$k$: Thermal Conductivity', fontsize=10)
            ax3.set_xlabel('Temperature (K)', fontsize=9)
            ax3.set_ylabel(r'$k$ $[W/(m·K)]$', fontsize=9)
            ax3.grid(True)

    # ZT 데이터 처리
    zt_keys = ['ZT', 'Z', 'PF']
    common_zt_samples = filter_common_samples_for_properties(dataframes, zt_keys)
    if selected_sample_id in common_zt_samples:
        df_ZT = process_zt_data(dataframes.get('ZT'), dataframes.get('Z'), dataframes.get('PF'))
        df_ZT = df_ZT[df_ZT['sample_id'] == selected_sample_id]
        if not df_ZT.empty:
            ax4.plot(df_ZT['temperature'], df_ZT['tepvalue'], marker='o', linestyle='-', color='b')
            ax4.set_title(r'$ZT$: Figure of Merit', fontsize=10)
            ax4.set_xlabel('Temperature (K)', fontsize=9)
            ax4.set_ylabel(r'$ZT$', fontsize=9)
            ax4.grid(True)

    plt.tight_layout()
    st.pyplot(fig)

# Main 함수 수정
def main():
    st.title("Thermoelectric Property Dashboard")

    # 파일 경로 리스트
    file_paths = {
        'alpha': 'df_alpha0.csv',
        'kappa': 'df_kappa0.csv',
        'sigma': 'df_sigma0.csv',
        'rho': 'df_rho0.csv',
        'ZT': 'df_zT0.csv',
        'Z': 'df_Z0.csv',
        'PF': 'df_PF0.csv'
    }

    # 데이터 로드
    dataframes = {}
    for key, file_path in file_paths.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            dataframes[key] = df
        else:
            st.warning(f"File {file_path} not found.")

    # 열전 물성 공통 sample_id 필터링
    required_keys = ['alpha', 'kappa', 'sigma', 'ZT']
    common_samples = filter_common_samples_for_properties(dataframes, required_keys)

    if not common_samples:
        st.error("No common sample IDs found across all properties.")
        return

    # 샘플 ID 선택
    selected_sample_id = st.sidebar.selectbox("Select Sample ID:", sorted(common_samples))

    # 데이터프레임 출력
    st.write("### Uploaded DataFrames")
    for key, df in dataframes.items():
        st.write(f"#### {key.capitalize()} DataFrame")
        st.dataframe(df)

    # 그래프 그리기
    st.write(f"### Graphs for Sample ID: {selected_sample_id}")
    create_and_plot_graphs(dataframes, selected_sample_id)

if __name__ == "__main__":
    main()

