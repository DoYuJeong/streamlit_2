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
    sample_id_sets = [set(df['sample_id'].unique()) for df in dataframes.values() if df is not None]
    return set.intersection(*sample_id_sets) if sample_id_sets else set()

# 업로드된 데이터 처리 함수
def load_and_process_data(uploaded_file):
    """
    업로드된 CSV 파일을 읽고 x, y 값을 리스트로 변환
    """
    try:
        df = pd.read_csv(uploaded_file, usecols=['prop_x', 'prop_y', 'x', 'y', 'sample_id'])
        df['x'] = df['x'].apply(ast.literal_eval)
        df['y'] = df['y'].apply(ast.literal_eval)
        return df
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
        return None

# 그래프 생성 함수
def create_and_plot_graphs(dataframes, selected_sample_id, is_uploaded=False):
    """
    데이터를 기반으로 그래프 생성 (기본 데이터와 업로드된 데이터 처리 방식 통합)
    """
    # 업로드 데이터 처리용 내부 함수
    def process_temperature(row):
        return [1 / t if t != 0 else np.nan for t in row['x']] if row['prop_x'] == 'Inverse temperature' else row['x']

    # 데이터 정리 함수
    def prepare_data(df, column_name, transform_func=None):
        if is_uploaded:
            lens = df['y'].map(len)
            sample_ids = df['sample_id'].repeat(lens)
            temperatures = np.concatenate(df.apply(process_temperature, axis=1).values)
            values = np.concatenate(df['y'].map(transform_func).values if transform_func else df['y'].values)
        else:
            sample_ids = df['sample_id']
            temperatures = df['temperature']
            values = df['tepvalue']
        
        return pd.DataFrame({'sample_id': sample_ids, 'temperature': temperatures, column_name: values})

    # 그래프 생성
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    keys = ["sigma", "alpha", "kappa", "ZT"]
    titles = ["Electrical Conductivity", "Seebeck Coefficient", "Thermal Conductivity", "ZT"]
    units = [r"$\sigma$ [S/cm]", r"$\alpha$ [$\mu V/K$]", r"$k$ [W/(m·K)]", r"ZT"]
    colors = ['m', 'g', 'r', 'b']
    
    for ax, key, title, unit, color in zip(axs.flatten(), keys, titles, units, colors):
        df = dataframes.get(key)
        if df is not None and not df.empty:
            if is_uploaded:
                filtered_df = df[(df['sample_id'] == selected_sample_id) & (df['prop_y'].isin([title]))]
                data = prepare_data(filtered_df, key)
            else:
                data = df[df['sample_id'] == selected_sample_id]
            if not data.empty:
                ax.plot(data['temperature'], data[key], marker='o', linestyle='-', color=color)
                ax.set_title(title)
                ax.set_xlabel("Temperature (K)")
                ax.set_ylabel(unit)
                ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)


# Streamlit 메인 함수
def main():
    st.title("Thermoelectric Property Dashboard")
    option = st.sidebar.radio("데이터 처리 옵션", ["기본 데이터 사용", "파일 업로드"])
    
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
        file_paths = {'sigma': 'df_combined_sigma.csv', 'alpha': 'df_alpha0.csv', 'kappa': 'df_kappa0.csv', 'ZT': 'df_combined_ZT.csv'}
        dataframes = {key: pd.read_csv(path) for key, path in file_paths.items()}
        common_sample_ids = get_common_sample_ids(dataframes)

        if common_sample_ids:
            selected_sample_id = st.sidebar.selectbox("Select Sample ID:", sorted(common_sample_ids))
            create_and_plot_graphs(dataframes, selected_sample_id)
        else:
            st.error("No common sample IDs with all properties found.")

    elif option == "파일 업로드":
        uploaded_file = st.sidebar.file_uploader("Thermoelectric Data File", type="csv")
        if uploaded_file:
            uploaded_df = load_and_process_data(uploaded_file)
            if uploaded_df is not None:
                sample_ids = uploaded_df['sample_id'].unique()
                selected_sample_id = st.sidebar.selectbox("Select Sample ID:", sorted(sample_ids))
                create_and_plot_graphs({'sigma': uploaded_df}, selected_sample_id, is_uploaded=True)

if __name__ == "__main__":
    main()

