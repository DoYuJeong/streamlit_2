import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

# 공통 sample_id 추출 함수
def get_common_sample_ids(dataframes):
    """
    여러 데이터프레임에서 공통 sample_id 추출
    """
    sample_id_sets = [set(df['sample_id']) for df in dataframes.values() if df is not None]
    return set.intersection(*sample_id_sets) if sample_id_sets else set()

# 그래프 생성 함수
def create_and_plot_graphs_filtered(dataframes, selected_sample_id):
    """
    특정 sample_id를 기준으로 필터링된 데이터를 사용해 그래프 생성
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    # Sigma 그래프
    df_sigma = dataframes.get('sigma')
    if df_sigma is not None:
        df_filtered_sigma = df_sigma[df_sigma['sample_id'] == selected_sample_id]
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
        df_filtered_kappa = df_kappa[df_kappa['sample_id'] == selected_sample_id]
        if not df_filtered_kappa.empty:
            df_filtered_kappa = df_filtered_kappa.sort_values(by='temperature')
            ax3.plot(df_filtered_kappa['temperature'], df_filtered_kappa['tepvalue'], marker='o', linestyle='-', color='r')
            ax3.set_title('Kappa')
            ax3.set_xlabel('Temperature (K)')
            ax3.set_ylabel(r'$k$ $[W/(m·K)]$')
            ax3.grid(True)

    # ZT 그래프
    df_ZT = dataframes.get('ZT')
    if df_ZT is not None:
        df_filtered_ZT = df_ZT[df_ZT['sample_id'] == selected_sample_id]
        if not df_filtered_ZT.empty:
            df_filtered_ZT = df_filtered_ZT.sort_values(by='temperature')
            ax4.plot(df_filtered_ZT['temperature'], df_filtered_ZT['tepvalue'], marker='o', linestyle='-', color='b')
            ax4.set_title('ZT')
            ax4.set_xlabel('Temperature (K)')
            ax4.set_ylabel(r'$ZT$')
            ax4.grid(True)

    plt.tight_layout()
    st.pyplot(fig)

# 데이터 로드 및 처리 함수
def load_and_process_data(uploaded_file, columns=['prop_x', 'prop_y', 'x', 'y', 'sample_id']):
    """
    업로드된 파일을 읽어 처리
    """
    try:
        df = pd.read_csv(uploaded_file, usecols=columns)
        df['x'] = df['x'].apply(ast.literal_eval)
        df['y'] = df['y'].apply(ast.literal_eval)
        return df
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
        return None

# Streamlit 메인 함수
def main():
    st.title("Thermoelectric Property Dashboard")

    st.sidebar.header("데이터 처리 방식 선택")
    option = st.sidebar.radio("데이터 처리 옵션", ["기본 데이터 사용", "파일 업로드"])

    dataframes = {}
    doi_df = None

    if option == "기본 데이터 사용":
        # 기본 파일 사용
        file_paths = {
            'sigma': 'df_combined_sigma.csv',
            'alpha': 'df_alpha0.csv',
            'kappa': 'df_kappa0.csv',
            'ZT': 'df_combined_ZT.csv'
        }
        doi_file = 'starrydata_papers_1.csv'

        for key, path in file_paths.items():
            try:
                dataframes[key] = pd.read_csv(path)
            except FileNotFoundError:
                st.warning(f"File {path} not found.")

        try:
            doi_df = pd.read_csv(doi_file, usecols=['SID', 'DOI', 'URL'])
        except FileNotFoundError:
            st.warning(f"DOI file {doi_file} not found.")

    elif option == "파일 업로드":
        # 파일 업로드 사용
        st.sidebar.subheader("데이터 파일 업로드")
        data_file = st.sidebar.file_uploader("Thermoelectric Data File", type="csv")
        doi_file = st.sidebar.file_uploader("DOI Data File", type="csv")

        if data_file:
            df = load_and_process_data(data_file)
            if df is not None:
                dataframes = {'ZT': df}
        
        if doi_file:
            doi_df = pd.read_csv(doi_file, usecols=['SID', 'DOI', 'URL'])

    # 공통 sample_id 추출 및 표시
    if dataframes:
        common_sample_ids = get_common_sample_ids(dataframes)
        if not common_sample_ids:
            st.error("No common sample IDs found across the datasets.")
            return

        selected_sample_id = st.sidebar.selectbox("Select Sample ID:", sorted(common_sample_ids))
        st.write(f"### Selected Sample ID: {selected_sample_id}")

        # 그래프 출력
        create_and_plot_graphs_filtered(dataframes, selected_sample_id)

        # DOI 정보 출력
        if doi_df is not None:
            doi_info = doi_df[doi_df['SID'] == selected_sample_id]
            if not doi_info.empty:
                st.write(f"**DOI**: {doi_info['DOI'].iloc[0]}")
                st.markdown(f"**URL**: [{doi_info['URL'].iloc[0]}]({doi_info['URL'].iloc[0]})")
            else:
                st.write("**DOI**: Not Available")
                st.write("**URL**: Not Available")

if __name__ == "__main__":
    main()

