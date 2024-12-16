import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 공통 sample_id 필터링 함수
def filter_common_samples(dataframes, required_keys=None):
    """
    dataframes: 데이터프레임 딕셔너리
    required_keys: 공통 샘플을 추출할 데이터프레임의 키 리스트
    """
    # 필터링에 필요한 데이터프레임만 선택
    if required_keys:
        dataframes = {key: dataframes[key] for key in required_keys if key in dataframes and dataframes[key] is not None}
    
    # 각 데이터프레임에서 sample_id 추출
    sample_sets = [set(df['sample_id']) for df in dataframes.values()]
    
    # 공통 sample_id 계산
    if sample_sets:
        common_samples = set.intersection(*sample_sets)
        return sorted(common_samples)
    return []

# 그래프 생성 함수
def create_and_plot_graphs(dataframes, common_samples):
    figsize = (10, 8)
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    # Sigma 데이터 처리 및 그래프 생성
    df_sigma = process_sigma_data(
        dataframes.get('sigma')[dataframes.get('sigma')['sample_id'].isin(common_samples)] if 'sigma' in dataframes else None,
        dataframes.get('rho')[dataframes.get('rho')['sample_id'].isin(common_samples)] if 'rho' in dataframes else None
    )
    if df_sigma is not None and not df_sigma.empty:
        ax1.plot(df_sigma['temperature'], df_sigma['tepvalue'], marker='o', linestyle='-', color='m')
        ax1.set_title(r'$\sigma$: Electrical Conductivity', fontsize=10)
        ax1.set_xlabel('Temperature (K)', fontsize=9)
        ax1.set_ylabel(r'$\sigma$ $[S/cm]$', fontsize=9)
        ax1.grid(True)

    # Alpha 데이터 그래프 생성
    df_alpha = dataframes.get('alpha')[dataframes.get('alpha')['sample_id'].isin(common_samples)] if 'alpha' in dataframes else None
    if df_alpha is not None and not df_alpha.empty:
        ax2.plot(df_alpha['temperature'], df_alpha['tepvalue'] * 1e6, marker='o', linestyle='-', color='g')
        ax2.set_title(r'$\alpha$: Seebeck Coefficient', fontsize=10)
        ax2.set_xlabel('Temperature (K)', fontsize=9)
        ax2.set_ylabel(r'$\alpha$ $[\mu V/K]$', fontsize=9)
        ax2.grid(True)

    # Kappa 데이터 그래프 생성
    df_kappa = dataframes.get('kappa')[dataframes.get('kappa')['sample_id'].isin(common_samples)] if 'kappa' in dataframes else None
    if df_kappa is not None and not df_kappa.empty:
        ax3.plot(df_kappa['temperature'], df_kappa['tepvalue'], marker='o', linestyle='-', color='r')
        ax3.set_title(r'$k$: Thermal Conductivity', fontsize=10)
        ax3.set_xlabel('Temperature (K)', fontsize=9)
        ax3.set_ylabel(r'$k$ $[W/(m·K)]$', fontsize=9)
        ax3.grid(True)

    # ZT 데이터 처리 및 그래프 생성
    df_ZT = process_zt_data(
        dataframes.get('ZT')[dataframes.get('ZT')['sample_id'].isin(common_samples)] if 'ZT' in dataframes else None,
        dataframes.get('Z')[dataframes.get('Z')['sample_id'].isin(common_samples)] if 'Z' in dataframes else None,
        dataframes.get('PF')[dataframes.get('PF')['sample_id'].isin(common_samples)] if 'PF' in dataframes else None
    )
    if df_ZT is not None and not df_ZT.empty:
        ax4.plot(df_ZT['temperature'], df_ZT['tepvalue'], marker='o', linestyle='-', color='b')
        ax4.set_title(r'$ZT$: Figure of Merit', fontsize=10)
        ax4.set_xlabel('Temperature (K)', fontsize=9)
        ax4.set_ylabel(r'$ZT$', fontsize=9)
        ax4.grid(True)

    plt.tight_layout()
    st.pyplot(fig)

# Streamlit 앱
def main():
    st.title("Thermoelectric Property Dashboard")

    # 데이터 업로드
    uploaded_files = st.file_uploader(
        "Upload your thermoelectric property files:",
        accept_multiple_files=True,
        type=["csv"]
    )

    if uploaded_files:
        # 데이터프레임 딕셔너리 생성
        dataframes = {}
        for uploaded_file in uploaded_files:
            property_name = uploaded_file.name.split('_')[1].split('0')[0]
            df = pd.read_csv(uploaded_file)
            dataframes[property_name] = df

        st.write("### Uploaded DataFrames")
        for key, df in dataframes.items():
            st.write(f"#### {key.capitalize()} DataFrame")
            st.dataframe(df)

        # 공통 sample_id 필터링
        required_keys = ['sigma', 'alpha', 'kappa', 'ZT']
        common_samples = filter_common_samples(dataframes, required_keys)
        if not common_samples:
            st.error("No common sample IDs found across selected properties.")
            return

        # 샘플 ID 선택
        sample_id = st.sidebar.selectbox("Select Sample ID:", common_samples)
        if sample_id:
            st.write(f"### Data for Sample ID: {sample_id}")
            sample_dataframes = {key: df[df['sample_id'] == sample_id] for key, df in dataframes.items()}
            create_and_plot_graphs(sample_dataframes, common_samples)

if __name__ == "__main__":
    main()

