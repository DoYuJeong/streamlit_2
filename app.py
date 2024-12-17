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

