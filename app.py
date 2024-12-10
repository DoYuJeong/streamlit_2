import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import os

# 데이터 로드 함수
def load_csv(file_path, usecols=None):
    try:
        return pd.read_csv(file_path, usecols=usecols)
    except Exception as e:
        st.error(f"Error loading file {file_path}: {e}")
        return None

# 열전 물성 데이터 처리 함수
def process_thermoelectric_data(df):
    def eval_columns(col):
        return col.apply(ast.literal_eval)

    try:
        df['x'] = eval_columns(df['x'])
        df['y'] = eval_columns(df['y'])
        return df
    except Exception as e:
        st.error(f"Error processing thermoelectric data: {e}")
        return None

# 공통 sample_id 필터링 함수
def filter_samples(df, property_mappings):
    property_samples = {
        key: set(df[df['prop_y'].isin(props)]['sample_id'])
        for key, (props, _) in property_mappings.items()
    }
    common_samples = set.intersection(*property_samples.values())
    filtered_df = df[df['sample_id'].isin(common_samples)]
    return filtered_df, sorted(common_samples)

# 시각화 및 데이터프레임 생성 함수
def create_and_plot_graphs(df, sample_id, property_mappings):
    def process_temperature(row):
        return [1 / t if t != 0 else np.nan for t in row['x']] if row['prop_x'] == 'Inverse temperature' else row['x']

    def create_property_df(filtered_df, column_name, transform_func=None):
        if filtered_df.empty:
            return pd.DataFrame(columns=['sample_id', 'temperature', column_name])

        lens = filtered_df['y'].map(len)
        temperatures = np.concatenate(filtered_df.apply(process_temperature, axis=1).values)
        values = np.concatenate(filtered_df['y'].map(transform_func).values if transform_func else filtered_df['y'].values)

        return pd.DataFrame({
            'sample_id': filtered_df['sample_id'].repeat(lens),
            'temperature': temperatures,
            column_name: values
        })

    # 데이터프레임 생성
    dataframes = {
        key: create_property_df(df[(df['prop_y'].isin(props)) & (df['sample_id'] == sample_id)], key, func)
        for key, (props, func) in property_mappings.items()
    }

    # 그래프 생성
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    keys = ['sigma', 'alpha', 'k', 'ZT']
    titles = ['Sigma', 'Alpha', 'Thermal Conductivity (k)', 'ZT']
    colors = ['m', 'g', 'r', 'b']

    for ax, key, title, color in zip(axs.flatten(), keys, titles, colors):
        df_key = dataframes.get(key)
        if df_key is not None and not df_key.empty:
            ax.plot(df_key['temperature'], df_key[key], marker='o', linestyle='-', color=color)
            ax.set_title(title)
            ax.set_xlabel('Temperature')
            ax.set_ylabel(key)
            ax.grid(True)

    plt.tight_layout()
    st.pyplot(fig)

    return dataframes

# Streamlit 앱
def main():
    st.title("Thermoelectric Property Dashboard")

    # 간단한 CV 추가
    st.markdown("""
    **Created by: Doyujeong**  
    **Email**: [doyujeong98@naver.com](mailto:doyujeong98@naver.com)  
    **GitHub**: [DoYuJeong](https://github.com/DoYuJeong)
    """)

    # 파일 경로 정의
    thermoelectric_file = 'starrydata_curves_1.csv'
    doi_file = 'starrydata_papers_1.csv'

    # 데이터 로드
    thermoelectric_df = load_csv(thermoelectric_file, usecols=['prop_x', 'prop_y', 'x', 'y', 'sample_id'])
    doi_df = load_csv(doi_file, usecols=['SID', 'DOI', 'URL'])

    if thermoelectric_df is not None and doi_df is not None:
        thermoelectric_df = process_thermoelectric_data(thermoelectric_df)
        merged_df = thermoelectric_df.merge(doi_df, left_on='sample_id', right_on='SID', how='left')

        # 공통 sample_id 필터링
        property_mappings = {
            'sigma': (['Electrical conductivity', 'Electrical resistivity'], None),
            'alpha': (['Seebeck coefficient', 'thermopower'], None),
            'k': (['Thermal conductivity', 'total thermal conductivity'], None),
            'ZT': (['ZT'], None)
        }
        filtered_df, common_samples = filter_samples(merged_df, property_mappings)

        if common_samples:
            sample_id = st.sidebar.selectbox("Select Sample ID:", common_samples)
            if sample_id:
                sample_data = filtered_df[filtered_df['sample_id'] == sample_id]
                st.write(f"### Data for Sample ID: {sample_id}")
                st.dataframe(sample_data)

                doi_info = sample_data[['DOI', 'URL']].drop_duplicates()
                if not doi_info.empty:
                    # DOI와 URL 값을 가져옵니다.
                    doi = doi_info['DOI'].iloc[0]
                    url = doi_info['URL'].iloc[0]
                
                    # DOI는 텍스트로 표시
                    st.write(f"**DOI**: {doi}")
                
                    # URL 데이터 자체에 하이퍼링크 연결
                    st.markdown(f"**URL**: [{url}]({url})")
                else:
                    st.write("**DOI**: Not Available")
                    st.write("**URL**: Not Available")

                st.write("### Graphs")
                create_and_plot_graphs(filtered_df, sample_id, property_mappings)
        else:
            st.error("No samples with all properties found.")
    else:
        st.error("Data loading failed.")

if __name__ == "__main__":
    main()
