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

# 기본 데이터 그래프 생성 함수
def create_and_plot_graphs_filtered(dataframes, selected_sample_id):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    # Sigma 그래프
    df_sigma = dataframes.get('sigma')
    if df_sigma is not None and not df_sigma.empty:
        df_filtered_sigma = df_sigma[df_sigma['sample_id'] == selected_sample_id]
        ax1.plot(df_filtered_sigma['temperature'], df_filtered_sigma['tepvalue'], marker='o', linestyle='-', color='m')
        ax1.set_title(r'$\sigma$: Electrical Conductivity', fontsize=10)
        ax1.set_xlabel('Temperature (K)', fontsize=9)
        ax1.set_ylabel(r'$\sigma$ $[S/cm]$', fontsize=9)
        ax1.grid(True)

    # Alpha 그래프
    df_alpha = dataframes.get('alpha')
    if df_alpha is not None and not df_alpha.empty:
        df_filtered_alpha = df_alpha[df_alpha['sample_id'] == selected_sample_id]
        ax2.plot(df_filtered_alpha['temperature'], df_filtered_alpha['tepvalue'] * 1e6, marker='o', linestyle='-', color='g')
        ax2.set_title(r'$\alpha$: Seebeck Coefficient', fontsize=10)
        ax2.set_xlabel('Temperature (K)', fontsize=9)
        ax2.set_ylabel(r'$\alpha$ $[\mu V/K]$', fontsize=9)
        ax2.grid(True)

    # kappa 그래프
    df_kappa = dataframes.get('kappa')
    if df_kappa is not None and not df_kappa.empty:
        df_filtered_kappa = df_kappa[df_kappa['sample_id'] == selected_sample_id]
        ax3.plot(df_filtered_kappa['temperature'], df_filtered_kappa['tepvalue'], marker='o', linestyle='-', color='r')
        ax3.set_title(r'$k$: Thermal Conductivity', fontsize=10)
        ax3.set_xlabel('Temperature (K)', fontsize=9)
        ax3.set_ylabel(r'$k$ $[W/(m·K)]$', fontsize=9)
        ax3.grid(True)

    # ZT 그래프
    df_ZT = dataframes.get('ZT')
    if df_ZT is not None and not df_ZT.empty:
        df_filtered_ZT = df_ZT[df_ZT['sample_id'] == selected_sample_id]
        ax4.plot(df_filtered_ZT['temperature'], df_filtered_ZT['tepvalue'], marker='o', linestyle='-', color='b')
        ax4.set_title(r'$ZT$: Figure of Merit', fontsize=10)
        ax4.set_xlabel('Temperature (K)', fontsize=9)
        ax4.set_ylabel(r'$ZT$', fontsize=9)
        ax4.grid(True)

    plt.tight_layout()
    st.pyplot(fig)

# 업로드 데이터 그래프 생성 함수
def create_and_plot_uploaded_graphs(uploaded_df, selected_sample_id):
    """
    업로드된 데이터에서 선택된 sample_id를 기준으로 4가지 열전 물성 그래프를 생성
    """

    # 온도 처리 함수
    def process_temperature(row):
        """
        prop_x 값이 'Inverse temperature'인 경우 1/T로 변환
        """
        return [1 / t if t != 0 else np.nan for t in row['x']] if row['prop_x'] == 'Inverse temperature' else row['x']
    
    # 데이터프레임 생성 함수
    def create_property_df(filtered_df, column_name, transform_func=None):
        """
        물성 데이터프레임 생성
        """
        if filtered_df.empty:
            return pd.DataFrame(columns=['sample_id', 'temperature', column_name])
        
        lens = filtered_df['y'].map(len)  # y 값의 길이 추출
        sample_ids = filtered_df['sample_id'].repeat(lens).values  # 반복된 sample_id
        temperatures = np.concatenate(filtered_df.apply(process_temperature, axis=1).values)  # 온도 데이터
        values = np.concatenate(filtered_df['y'].map(transform_func).values if transform_func else filtered_df['y'].values)

        return pd.DataFrame({
            'sample_id': sample_ids,
            'temperature': temperatures,
            column_name: values
        }).sort_values(by='temperature').reset_index(drop=True)

    # 열전 물성 매핑
    property_mappings = {
        'sigma': (['Electrical conductivity', 'Electrical resistivity'], lambda y: [1 / v if v != 0 else np.nan for v in y]),
        'alpha': (['Seebeck coefficient', 'thermopower'], None),
        'k': (['Thermal conductivity', 'total thermal conductivity'], None),
        'ZT': (['ZT'], None)
    }

    # 데이터프레임 생성
    dataframes = {}
    for key, (properties, transform_func) in property_mappings.items():
        filtered_df = uploaded_df[(uploaded_df['prop_y'].isin(properties)) & (uploaded_df['sample_id'] == selected_sample_id)]
        dataframes[key] = create_property_df(filtered_df, key, transform_func)

    # 그래프 생성
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    # Sigma 그래프
    df_sigma = dataframes.get('sigma')
    if df_sigma is not None and not df_sigma.empty:
        ax1.plot(df_sigma['temperature'], df_sigma['sigma'], marker='o', linestyle='-', color='m')
        ax1.set_title(r'$\sigma$: Electrical Conductivity', fontsize=10)
        ax1.set_xlabel('Temperature (K)', fontsize=9)
        ax1.set_ylabel(r'$\sigma$ $[S/cm]$', fontsize=9)
        ax1.grid(True)

    # Alpha 그래프
    df_alpha = dataframes.get('alpha')
    if df_alpha is not None and not df_alpha.empty:
        ax2.plot(df_alpha['temperature'], df_alpha['alpha'] * 1e6, marker='o', linestyle='-', color='g')
        ax2.set_title(r'$\alpha$: Seebeck Coefficient', fontsize=10)
        ax2.set_xlabel('Temperature (K)', fontsize=9)
        ax2.set_ylabel(r'$\alpha$ $[\mu V/K]$', fontsize=9)
        ax2.grid(True)

    # kappa 그래프
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


# 데이터 로드 및 처리 함수
def load_and_process_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, usecols=['prop_x', 'prop_y', 'x', 'y', 'sample_id'])
        df['x'] = df['x'].apply(ast.literal_eval)
        df['y'] = df['y'].apply(ast.literal_eval)
        return df
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
        return None

# Streamlit 메인 함수
def main():
    st.title("Thermoelectric Property Dashboard")

    # 간단한 CV 추가
    st.markdown("""
    **Created by: Doyujeong**  
    **Email**: [doyujeong98@naver.com](mailto:doyujeong98@naver.com)  
    **GitHub**: [DoYuJeong](https://github.com/DoYuJeong)
    """)

    st.markdown("""
    ### 📊 **이 대시보드는 무엇인가요?**  
    이 대시보드는 **열전재료**의 주요 물성을 시각화하는 도구입니다.  
    아래의 물성을 온도에 따라 그래프로 확인할 수 있습니다:  
    - **Sigma**: 전기전도도 (Electrical Conductivity)  
    - **Alpha**: 제벡계수 (Seebeck Coefficient)  
    - **Kappa**: 열전도도 (Thermal Conductivity)  
    - **ZT**: 열전 성능 지수 (Figure of Merit)  

    ---

    ### 📝 **사용 방법**  
    1. **왼쪽 사이드바에서 샘플 ID를 선택하세요.**  
       - 샘플 ID는 특정 재료의 데이터 세트를 의미합니다.  
    
    2. **그래프 확인하기**  
       - 선택한 샘플 ID에 대한 **온도별 열전 물성 그래프**를 확인할 수 있습니다.  
    
    3. **데이터 테이블 보기**  
       - 그래프에 사용된 **원본 데이터**를 테이블 형식으로 제공합니다.  
    
    4. **연구 논문 정보 확인**  
       - 해당 샘플 ID와 관련된 **DOI 및 URL** 링크를 통해 논문 정보를 확인할 수 있습니다.  
    """)
    
    st.sidebar.header("데이터 처리 방식 선택")
    option = st.sidebar.radio("데이터 처리 옵션", ["기본 데이터 사용", "파일 업로드"])

    dataframes = {}
    doi_df = None

    st.sidebar.header("데이터 처리 방식 선택")
    option = st.sidebar.radio("데이터 처리 옵션", ["기본 데이터 사용", "파일 업로드"])

    if option == "기본 데이터 사용":
        # 기본 파일 사용
        file_paths = {
            'sigma': 'df_combined_sigma.csv',
            'alpha': 'df_alpha0.csv',
            'kappa': 'df_kappa0.csv',
            'ZT': 'df_combined_ZT.csv'
        }
        doi_file = 'starrydata_papers_1.csv'

        dataframes = {key: pd.read_csv(path) for key, path in file_paths.items()}
        common_sample_ids = get_common_sample_ids(dataframes)

        # 샘플 ID 선택 및 그래프
        selected_sample_id = st.sidebar.selectbox("Select Sample ID:", sorted(common_sample_ids))
        create_and_plot_graphs_filtered(dataframes, selected_sample_id)

        try:
            doi_df = pd.read_csv(doi_file, usecols=['SID', 'DOI', 'URL'])
        except FileNotFoundError:
            st.warning(f"DOI file {doi_file} not found.")

    elif option == "파일 업로드":
        # 파일 업로드
        uploaded_file = st.sidebar.file_uploader("Thermoelectric Data File", type="csv")
        doi_file = st.sidebar.file_uploader("DOI Data File", type="csv")

        if uploaded_file:
            uploaded_df = load_and_process_data(uploaded_file)
            doi_df = load_csv(doi_file, usecols=['SID', 'DOI', 'URL']) if doi_file else None

            if uploaded_df is not None:
                sample_ids = uploaded_df['sample_id'].unique()
                selected_sample_id = st.sidebar.selectbox("Select Sample ID:", sample_ids)
                create_and_plot_uploaded_graphs(uploaded_df, selected_sample_id, doi_df)

    
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

        # 정확한 데이터프레임 출력
        if 'sigma' in dataframes and not dataframes['sigma'].empty:
            df_sigma_filtered = dataframes['sigma'][dataframes['sigma']['sample_id'] == selected_sample_id]
            st.write("#### Electrical conductivity DataFrame")
            st.dataframe(df_sigma_filtered)
    
        if 'alpha' in dataframes and not dataframes['alpha'].empty:
            df_alpha_filtered = dataframes['alpha'][dataframes['alpha']['sample_id'] == selected_sample_id]
            st.write("#### Seebeck coefficient DataFrame")
            st.dataframe(df_alpha_filtered)
    
        if 'kappa' in dataframes and not dataframes['kappa'].empty:
            df_kappa_filtered = dataframes['kappa'][dataframes['kappa']['sample_id'] == selected_sample_id]
            st.write("#### Thermal conductivity DataFrame")
            st.dataframe(df_kappa_filtered)
    
        if 'ZT' in dataframes and not dataframes['ZT'].empty:
            df_ZT_filtered = dataframes['ZT'][dataframes['ZT']['sample_id'] == selected_sample_id]
            st.write("#### ZT DataFrame")
            st.dataframe(df_ZT_filtered)


if __name__ == "__main__":
    main()

