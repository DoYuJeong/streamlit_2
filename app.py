import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

# 데이터 로드 및 처리 함수
def load_and_process_data(uploaded_file):
    def eval_columns(col):
        try:
            return col.apply(ast.literal_eval)
        except Exception as e:
            st.error(f"Error parsing column values: {e}")
            return col

    try:
        df = pd.read_csv(uploaded_file, usecols=['prop_x', 'prop_y', 'x', 'y', 'sample_id'])
        df['x'] = eval_columns(df['x'])
        df['y'] = eval_columns(df['y'])
        return df
    except Exception as e:
        st.error(f"Error loading thermoelectric data file: {e}")
        return None

def load_doi_data(doi_file):
    try:
        doi_df = pd.read_csv(doi_file, usecols=['SID', 'DOI', 'URL'])
        return doi_df
    except Exception as e:
        st.error(f"Error loading DOI data file: {e}")
        return None

def merge_thermoelectric_with_doi(thermoelectric_df, doi_df):
    try:
        return thermoelectric_df.merge(doi_df, left_on='sample_id', right_on='SID', how='left')
    except Exception as e:
        st.error(f"Error merging data: {e}")
        return thermoelectric_df

# 그래프 그리기 함수
def plot_TEP(df, sample_id):
    property_mappings = {
        'sigma': 'Electrical conductivity',
        'alpha': 'Seebeck coefficient',
        'k': 'Thermal conductivity',
        'ZT': 'ZT'
    }

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
    axes = [ax1, ax2, ax3, ax4]

    for i, (prop, title) in enumerate(property_mappings.items()):
        filtered_df = df[(df['prop_y'] == title) & (df['sample_id'] == sample_id)]
        if not filtered_df.empty:
            temperatures = np.concatenate(filtered_df['x'].values)
            values = np.concatenate(filtered_df['y'].values)
            axes[i].plot(temperatures, values, marker='o')
            axes[i].set_title(title)
            axes[i].set_xlabel("Temperature (K)")
            axes[i].set_ylabel(title)

    plt.tight_layout()
    st.pyplot(fig)

# 메인 함수
def main():
    st.title("Thermoelectric Property Dashboard")

    # 사이드바: 데이터 선택
    st.sidebar.header("데이터 옵션")
    option = st.sidebar.radio("데이터 처리 방식 선택", ["파일 업로드", "기존 데이터 사용"])

    # 파일 업로드를 통한 처리
    if option == "파일 업로드":
        st.subheader("1️⃣ 파일 업로드를 통한 데이터 처리")
        data_file = st.sidebar.file_uploader("Thermoelectric 데이터 파일 업로드", type="csv")
        doi_file = st.sidebar.file_uploader("DOI 데이터 파일 업로드", type="csv")

        if data_file and doi_file:
            thermoelectric_df = load_and_process_data(data_file)
            doi_df = load_doi_data(doi_file)

            if thermoelectric_df is not None and doi_df is not None:
                merged_df = merge_thermoelectric_with_doi(thermoelectric_df, doi_df)
                sample_ids = merged_df['sample_id'].unique()
                selected_sample_id = st.sidebar.selectbox("Select Sample ID", sample_ids)

                st.write(f"### 선택된 Sample ID: {selected_sample_id}")
                plot_TEP(merged_df, selected_sample_id)

    # 기존 데이터 사용
    elif option == "기존 데이터 사용":
        st.subheader("2️⃣ 기본 파일을 사용한 데이터 처리")

        # 기본 파일 경로 설정
        file_paths = {
            'sigma': 'df_combined_sigma.csv',
            'alpha': 'df_alpha0.csv',
            'kappa': 'df_kappa0.csv',
            'ZT': 'df_combined_ZT.csv'
        }
        doi_file = 'starrydata_papers_1.csv'

        # 데이터 로드
        dataframes = {key: pd.read_csv(path) for key, path in file_paths.items() if path}
        doi_df = pd.read_csv(doi_file, usecols=['SID', 'DOI', 'URL'])

        # 공통 Sample ID 찾기
        common_samples = set.intersection(*[set(df['sample_id']) for df in dataframes.values()])
        selected_sample_id = st.sidebar.selectbox("Select Sample ID", sorted(common_samples))

        # 그래프 출력
        st.write(f"### 선택된 Sample ID: {selected_sample_id}")
        combined_df = pd.concat(dataframes.values())
        plot_TEP(combined_df, selected_sample_id)

        # DOI 정보 출력
        doi_info = doi_df[doi_df['SID'] == selected_sample_id]
        if not doi_info.empty:
            st.write(f"**DOI**: {doi_info['DOI'].iloc[0]}")
            st.markdown(f"**URL**: [{doi_info['URL'].iloc[0]}]({doi_info['URL'].iloc[0]})")
        else:
            st.write("**DOI**: Not Available")
            st.write("**URL**: Not Available")

if __name__ == "__main__":
    main()
