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

def create_and_plot_graphs(df, sample_id, property_mappings):
    def process_temperature(row):
        return [1 / t if t != 0 else np.nan for t in row['x']] if row['prop_x'] == 'Inverse temperature' else row['x']
    
    def create_property_df(filtered_df, column_name, transform_func=None):
        if filtered_df.empty:
            return pd.DataFrame(columns=['sample_id', 'temperature', column_name])
        
        lens = filtered_df['y'].map(len)
        sample_ids = filtered_df['sample_id'].repeat(lens)
        temperatures = np.concatenate(filtered_df.apply(process_temperature, axis=1).values)
        values = np.concatenate(filtered_df['y'].map(transform_func).values if transform_func else filtered_df['y'].values)
        
        df = pd.DataFrame({
            'sample_id': sample_ids,
            'temperature': temperatures,
            column_name: values
        })
        return df.sort_values(by='temperature').reset_index(drop=True)
    
    # 데이터프레임 생성
    dataframes = {
        key: create_property_df(df[(df['prop_y'].isin(properties)) & (df['sample_id'] == sample_id)], key, func)
        for key, (properties, func) in property_mappings.items()
    }
    
    # 개별 그래프 그리기
    figsize = (10, 8)
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
    
    # Sigma 그래프
    df_sigma = dataframes.get('sigma')
    if df_sigma is not None and not df_sigma.empty:
        ax1.plot(df_sigma['temperature'], df_sigma['sigma'], marker='o', linestyle='-', color='m')
        ax1.set_title(r'$\sigma$: Electrical Conductivity', fontsize=10)
        ax1.set_xlabel('Temperature (K)', fontsize=9)
        ax1.set_ylabel(r'$\sigma$ $[S/cm]$', fontsize=9)
        ax1.grid(True)
    
    # Alpha 그래프 (Y축 데이터에 1e6 곱하기)
    df_alpha = dataframes.get('alpha')
    if df_alpha is not None and not df_alpha.empty:
        ax2.plot(df_alpha['temperature'], df_alpha['alpha'] * 1e6, marker='o', linestyle='-', color='g')
        ax2.set_title(r'$\alpha$: Seebeck Coefficient', fontsize=10)
        ax2.set_xlabel('Temperature (K)', fontsize=9)
        ax2.set_ylabel(r'$\alpha$ $[\mu V/K]$', fontsize=9)
        ax2.grid(True)
    
    # k 그래프
    df_k = dataframes.get('k')
    if df_k is not None and not df_k.empty:
        ax3.plot(df_k['temperature'], df_k['k'], marker='o', linestyle='-', color='r')
        ax3.set_title(r'$k$: Thermal Conductivity', fontsize=10)
        ax3.set_xlabel('Temperature (K)', fontsize=9)
        ax3.set_ylabel(r'$k$ $[W/(m·K)]$', fontsize=9)
        ax3.grid(True)
    
    # ZT 그래프
    df_ZT = dataframes.get('ZT')
    if df_ZT is not None and not df_ZT.empty:
        ax4.plot(df_ZT['temperature'], df_ZT['ZT'], marker='o', linestyle='-', color='b')
        ax4.set_title(r'$ZT$: Figure of Merit', fontsize=10)
        ax4.set_xlabel('Temperature (K)', fontsize=9)
        ax4.set_ylabel(r'$ZT$', fontsize=9)
        ax4.grid(True)
    
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
                    doi = doi_info['DOI'].iloc[0]
                    url = doi_info['URL'].iloc[0]
                    st.write(f"**DOI**: {doi}")
                    st.markdown(f"**URL**: [{url}]({url})")
                else:
                    st.write("**DOI**: Not Available")
                    st.write("**URL**: Not Available")

                st.write("### Graphs")
                dataframes = create_and_plot_graphs(filtered_df, sample_id, property_mappings)

                # 데이터프레임 출력
                st.write("### DataFrames for Each Property")
                
                # 키 이름에 대한 풀네임 매핑
                property_fullnames = {
                    'sigma': 'Electrical Conductivity',
                    'alpha': 'Seebeck Coefficient',
                    'k': 'Thermal Conductivity',
                    'ZT': 'Figure of Merit'
                }
                
                # 데이터프레임 출력 루프
                for key, df in dataframes.items():
                    if key in property_fullnames:  # 키에 대한 풀네임이 있는 경우
                        st.write(f"#### {property_fullnames[key]}")
                    else:  # 매핑이 없는 경우 기본 키 사용
                        st.write(f"#### {key.capitalize()} DataFrame")
                    st.dataframe(df)
        else:
            st.error("No samples with all properties found.")
    else:
        st.error("Data loading failed.")

if __name__ == "__main__":
    main()
