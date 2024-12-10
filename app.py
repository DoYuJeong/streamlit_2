import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import os

# 데이터 로드 및 처리 함수
def load_and_process_data(file_path):
    def eval_columns(col):
        try:
            return col.apply(ast.literal_eval)
        except Exception as e:
            st.error(f"Error parsing column values: {e}")
            return col

    try:
        # Thermoelectric 데이터 파일 읽기
        df = pd.read_csv(file_path, usecols=['prop_x', 'prop_y', 'x', 'y', 'sample_id'])
        df['x'] = eval_columns(df['x'])
        df['y'] = eval_columns(df['y'])
        return df
    except Exception as e:
        st.error(f"Error loading thermoelectric data file: {e}")
        return None

# DOI 데이터를 로드하는 함수
def load_doi_data(file_path):
    try:
        # DOI 데이터 파일 읽기
        doi_df = pd.read_csv(file_path, usecols=['SID', 'DOI', 'URL'])
        return doi_df
    except Exception as e:
        st.error(f"Error loading DOI data file: {e}")
        return None

# Streamlit 앱
def main():
    st.title("Thermoelectric Property Dashboard")

    # 파일 경로 정의
    thermoelectric_file = 'starrydata_curves_1.csv'
    doi_file = 'starrydata_papers_1.csv'

    # 파일 존재 여부 확인
    if not os.path.exists(thermoelectric_file):
        st.error(f"Thermoelectric data file '{thermoelectric_file}' not found.")
        return

    if not os.path.exists(doi_file):
        st.error(f"DOI data file '{doi_file}' not found.")
        return

    # 데이터 로드
    thermoelectric_df = load_and_process_data(thermoelectric_file)
    doi_df = load_doi_data(doi_file)

    if thermoelectric_df is not None and doi_df is not None:
        # Thermoelectric 데이터와 DOI 데이터 병합
        merged_df = thermoelectric_df.merge(doi_df, left_on='sample_id', right_on='SID', how='left')
        st.write("Data loaded and merged successfully!")

        # 열전 물성이 모두 존재하는 샘플 필터링
        property_mappings = {
            'sigma': (['Electrical conductivity', 'Electrical resistivity'], None),
            'alpha': (['Seebeck coefficient', 'thermopower'], None),
            'k': (['Thermal conductivity', 'total thermal conductivity'], None),
            'ZT': (['ZT'], None),
        }

        filtered_df, common_samples = filter_samples_with_all_properties(merged_df, property_mappings)

        if not common_samples:
            st.error("No samples with all thermoelectric properties found!")
            return

        # 공통 샘플 ID만 표시
        sample_id = st.sidebar.selectbox("Select Sample ID (with all properties):", sorted(common_samples))

        if sample_id:
            # 선택한 샘플 ID 데이터 필터링
            sample_data = filtered_df[filtered_df['sample_id'] == sample_id]
            st.write(f"### Data Table for Sample ID: {sample_id}")
            st.dataframe(sample_data)

            # DOI 정보 출력
            doi_info = sample_data[['DOI', 'URL']].drop_duplicates()
            if not doi_info.empty:
                doi = doi_info['DOI'].iloc[0]
                url = doi_info['URL'].iloc[0]

                # DOI를 링크로 표시
                st.write(f"**DOI**: [Link]({doi})")

                # URL을 클릭 가능한 링크로 표시
                st.markdown(f"**URL**: [Visit Here]({url})")
            else:
                st.write("**DOI**: Not Available")
                st.write("**URL**: Not Available")

            # 그래프 그리기를 위한 데이터프레임 생성
            dataframes = plot_TEP(filtered_df, sample_id, return_dataframes=True)

            # 전처리된 데이터프레임 출력
            st.write(f"### Processed Data for Sample ID: {sample_id}")

            if dataframes:
                df_sigma = dataframes.get('sigma', pd.DataFrame())
                df_alpha = dataframes.get('alpha', pd.DataFrame())
                df_k = dataframes.get('k', pd.DataFrame())
                df_ZT = dataframes.get('ZT', pd.DataFrame())

                st.write("#### Sigma")
                st.dataframe(df_sigma)

                st.write("#### Alpha")
                st.dataframe(df_alpha)

                st.write("#### Thermal Conductivity (k)")
                st.dataframe(df_k)

                st.write("#### ZT")
                st.dataframe(df_ZT)

            # 그래프 그리기
            st.write("### Property Graphs")
            plot_TEP(filtered_df, sample_id)
    else:
        st.info("Failed to load data.")

if __name__ == "__main__":
    main()
