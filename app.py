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

# Thermoelectric 데이터와 DOI 데이터를 병합하는 함수
def merge_thermoelectric_with_doi(thermoelectric_df, doi_df):
    try:
        # 병합: sample_id(SID)를 기준으로 DOI 데이터 연결
        merged_df = thermoelectric_df.merge(doi_df, left_on='sample_id', right_on='SID', how='left')
        return merged_df
    except Exception as e:
        st.error(f"Error merging data: {e}")
        return thermoelectric_df
        
# 열전 물성이 모두 존재하는 샘플 필터링 함수
def filter_samples_with_all_properties(df, property_mappings):
    property_samples = {}

    # 각 물성별 샘플 ID를 추출
    for prop_key, (properties, _) in property_mappings.items():
        property_samples[prop_key] = df[df['prop_y'].isin(properties)]['sample_id'].unique()

    # 공통 샘플 ID를 계산
    common_samples = set.intersection(*[set(samples) for samples in property_samples.values()])
    return df[df['sample_id'].isin(common_samples)], common_samples

# TEP 그래프 그리기 함수
def plot_TEP(df, sample_id):
    def process_temperature(row):
        if row['prop_x'] == 'Inverse temperature':
            return [1/t if t != 0 else np.nan for t in row['x']]
        return row['x']

    def create_property_df(filtered_df, column_name, transform_func=None):
        if filtered_df.empty:
            return pd.DataFrame(columns=['sample_id', 'temperature', column_name])

        lens = filtered_df['y'].map(len)
        sample_ids = filtered_df['sample_id'].repeat(lens).values
        temperatures = np.concatenate(filtered_df.apply(process_temperature, axis=1).values)
        values = np.concatenate(filtered_df.apply(lambda row: transform_func(row) if transform_func else row['y'], axis=1).values)

        return pd.DataFrame({
            'sample_id': sample_ids,
            'temperature': temperatures,
            column_name: values
        })

    property_mappings = {
        'sigma': (
            ['Electrical conductivity', 'Electrical resistivity'], 
            lambda row: [1/v if v != 0 else np.nan for v in row['y']] if row['prop_y'] == 'Electrical resistivity' else row['y']
        ),
        'alpha': (
            ['Seebeck coefficient', 'thermopower'], 
            None
        ),
        'k': (
            ['Thermal conductivity', 'total thermal conductivity'], 
            None
        ),
        'ZT': (
            ['ZT'], 
            None
        )
    }

    def create_property_dataframes(df, sample_id, property_mappings):
        dataframes = {}
        for column_name, (properties, transform_func) in property_mappings.items():
            filtered_df = df[(df['prop_y'].isin(properties)) & (df['sample_id'] == sample_id)]
            dataframes[column_name] = create_property_df(filtered_df, column_name, transform_func).sort_values(by='temperature')
        return dataframes

    dataframes = create_property_dataframes(df, sample_id, property_mappings)

    df_sigma = dataframes['sigma']
    df_alpha = dataframes['alpha']
    df_k = dataframes['k']
    df_ZT = dataframes['ZT']

    # 그래프 그리기
    figsize = (10, 8)
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    if not df_sigma.empty:
        ax1.plot(df_sigma['temperature'], df_sigma['sigma'], marker='o', linestyle='-', color='m')
        ax1.set_title('Sigma')
        ax1.set_xlabel('Temperature')
        ax1.set_ylabel(r'$\sigma$ $[S/cm]$')
        ax1.grid(True)

    if not df_alpha.empty:
        ax2.plot(df_alpha['temperature'], df_alpha['alpha'] * 1e6, marker='o', linestyle='-', color='g')
        ax2.set_title('Alpha')
        ax2.set_xlabel('Temperature')
        ax2.set_ylabel(r'$\alpha$ $[\mu V/K]$')
        ax2.grid(True)

    if not df_k.empty:
        ax3.plot(df_k['temperature'], df_k['k'], marker='o', linestyle='-', color='r')
        ax3.set_title('K')
        ax3.set_xlabel('Temperature')
        ax3.set_ylabel(r'$k$ $[W/(m·K)]$')
        ax3.grid(True)

    if not df_ZT.empty:
        ax4.plot(df_ZT['temperature'], df_ZT['ZT'], marker='o', linestyle='-', color='b')
        ax4.set_title(r'$ZT$')
        ax4.set_xlabel('Temperature')
        ax4.set_ylabel(r'$ZT$')
        ax4.grid(True)

    plt.tight_layout()
    st.pyplot(fig)  # Streamlit에서 그래프 표시

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
