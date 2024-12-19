import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
    # 그래프 영역 설정
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

# 업로드 데이터 그래프 생성 함수
def create_and_plot_graphs(dataframes, selected_sample_id):
    def process_temperature(row):
        return [1 / t if t != 0 else np.nan for t in row['x']] if row['prop_x'] == 'Inverse temperature' else row['x']

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    titles = ["Electrical Conductivity", "Seebeck Coefficient", "Thermal Conductivity", "ZT"]
    units = [r"$\sigma$ [S/cm]", r"$\alpha$ [$\mu V/K$]", r"$k$ [W/(m·K)]", r"ZT"]
    colors = ['m', 'g', 'r', 'b']
    keys = ["sigma", "alpha", "kappa", "ZT"]

    for ax, key, title, unit, color in zip(axs.flatten(), keys, titles, units, colors):
        df = dataframes.get(key)
        if df is not None and not df.empty:
            filtered_df = df[df['sample_id'] == selected_sample_id]
            lens = filtered_df['y'].map(len)
            temperatures = np.concatenate(filtered_df.apply(process_temperature, axis=1).values)
            values = np.concatenate(filtered_df['y'].values)

            ax.plot(temperatures, values, marker='o', linestyle='-', color=color)
            ax.set_title(title)
            ax.set_xlabel("Temperature (K)")
            ax.set_ylabel(unit)
            ax.grid(True)

    plt.tight_layout()
    st.pyplot(fig)

# 데이터 로드 및 처리 함수
def load_and_process_data(uploaded_file):
    def eval_columns(col):
        try:
            return col.apply(ast.literal_eval)
        except Exception as e:
            st.error(f"Error parsing column values: {e}")
            return col

    try:
        # Thermoelectric 데이터 파일 읽기
        df = pd.read_csv(uploaded_file, usecols=['prop_x', 'prop_y', 'x', 'y', 'sample_id'])
        df['x'] = eval_columns(df['x'])
        df['y'] = eval_columns(df['y'])
        return df
    except Exception as e:
        st.error(f"Error loading thermoelectric data file: {e}")
        return None

# DOI 데이터를 로드하는 함수
def load_doi_data(doi_file):
    try:
        # DOI 데이터 파일 읽기
        doi_df = pd.read_csv(doi_file, usecols=['SID', 'DOI', 'URL'])
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

    return dataframes


# Streamlit 메인 함수
def main():
    st.title("Thermoelectric Property Dashboard")
    option = st.sidebar.radio("데이터 처리 방식", ["기본 데이터 사용", "파일 업로드"])
    
    # 간단한 CV 추가
    st.markdown("""
    **Created by: Doyujeong**  
    **Email**: [doyujeong98@naver.com](mailto:doyujeong98@naver.com)  
    **GitHub**: [https://github.com/DoYuJeong]
    """)

    # 탭 생성
    tabs = st.tabs(["📊 Dashboard Overview", "📂 Data Frames", "📈 Graphs"])

    # 탭 1: Dashboard Overview
    with tabs[0]:
        st.markdown("""
            이 대시보드는 **한국전기연구원**에서 진행 중인 열전 물성 데이터를 분석하고 시각화하는 인턴 프로젝트의 일부입니다.  
            본 프로젝트는 **Starrydata**에서 제공하는 열전 물성 데이터를 활용하여, 열전재료의 성능 특성을 온도에 따라 시각적으로 이해하는 것을 목표로 합니다.  
            아래의 물성을 온도에 따라 그래프로 확인할 수 있습니다:  
            - **Sigma**: 전기전도도 (Electrical Conductivity)  
            - **Alpha**: 제벡계수 (Seebeck Coefficient)  
            - **Kappa**: 열전도도 (Thermal Conductivity)  
            - **ZT**: 열전 성능 지수 (Figure of Merit)  
        
            #### **프로젝트 목표**  
            - 열전 물성이 모두 존재하는 샘플을 필터링  
            - 전처리된 데이터를 활용한 그래프 생성  
            - Streamlit 기반의 대시보드를 통해 데이터 시각화  

            #### **데이터 출처**  
            - 데이터는 [Starrydata](https://www.starrydata.org/)에서 다운로드했습니다.  
            - Starrydata는 열전 물성 데이터를 공유하는 오픈 데이터 플랫폼으로, 다양한 재료의 특성을 제공합니다.
            
            ---
        
            ### 📝 **사용 방법**  
            
            #### **1. 데이터 처리 방식 선택**  
            왼쪽 사이드바에서 원하는 **데이터 처리 방식**을 선택하세요:  
            - **기본 데이터 사용**: 이미 준비된 데이터를 사용하여 열전 물성을 확인합니다.  
            - **파일 업로드**: 사용자가 보유한 CSV 데이터를 업로드하여 열전 물성을 확인합니다.  
        
            ---
        
            #### **2. 샘플 ID 선택하기**  
            - 선택 가능한 **샘플 ID**는 특정 재료의 데이터 세트를 의미합니다.  
            - **기본 데이터 사용**: 미리 준비된 데이터에 포함된 샘플 ID가 자동으로 제공됩니다.  
            - **파일 업로드**: 업로드한 데이터에서 공통 열전 물성이 모두 존재하는 샘플 ID를 자동으로 탐색합니다.  
        
            ---
        
            #### **3. 그래프 확인하기**  
            - 선택한 샘플 ID에 대한 **온도별 열전 물성 그래프**를 확인할 수 있습니다.  
            - **4가지 열전 물성**을 시각화합니다:  
        
            ---
        
            #### **4. 데이터 테이블 보기**  
            - 그래프에 사용된 **전처리된 데이터**를 테이블 형식으로 제공합니다.  
            - 각 열전 물성별로 정리된 데이터 (`Sigma`, `Alpha`, `Kappa`, `ZT`)를 확인할 수 있습니다. 
                
            ---
        
            #### **5. 연구 논문 정보 확인**  
            - 선택한 샘플 ID와 관련된 **DOI**(연구 논문 정보) 및 **URL** 링크를 확인할 수 있습니다.  
            - 논문 링크를 통해 해당 데이터를 연구에 활용한 정보를 찾아볼 수 있습니다.  
            """)

    if option == "기본 데이터 사용":
        # 데이터 로드
        file_paths = {
            'sigma': 'df_combined_sigma.csv',
            'alpha': 'df_alpha0.csv',
            'kappa': 'df_kappa0.csv',
            'ZT': 'df_combined_ZT.csv'
        }
        doi_file = 'starrydata_papers_1.csv'
        
        # 데이터프레임 딕셔너리 구성
        dataframes = {}
        for key, path in file_paths.items():
            try:
                dataframes[key] = pd.read_csv(path)
            except FileNotFoundError:
                st.warning(f"File {path} not found.")
    
        # DOI 데이터 불러오기
        try:
            doi_df = pd.read_csv(doi_file, usecols=['SID', 'DOI', 'URL'])
        except FileNotFoundError:
            st.error(f"DOI file {doi_file} not found.")
            return
        
        
        # 공통 sample_id 추출
        common_sample_ids = get_common_sample_ids(dataframes)
    
        if not common_sample_ids:
            st.error("No common sample IDs found across all datasets.")
            return
    
        # 사용자 선택: sample_id
        selected_sample_id = st.sidebar.selectbox("Select Sample ID:", sorted(common_sample_ids))

        # 탭 2: Data Frames
        with tabs[1]:
            st.write(f"### Selected Sample ID: {selected_sample_id}")

            # DOI 정보 출력
            doi_info = doi_df[doi_df['SID'] == selected_sample_id]
            if not doi_info.empty:
                doi = doi_info['DOI'].iloc[0]
                url = doi_info['URL'].iloc[0]
                st.write(f"**DOI**: {doi}")
                st.markdown(f"**URL**: [{url}]({url})")
            else:
                st.write("**DOI**: Not Available")
                st.write("**URL**: Not Available")

            # 데이터프레임 출력
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

        # 탭 3: Graphs
        with tabs[2]:
            st.write("### Graphs for Selected Sample ID")
            create_and_plot_graphs_filtered(dataframes, selected_sample_id)

    elif option == "파일 업로드":
        # 사이드바: 파일 업로드
        st.sidebar.header("Upload Files")
        data_file = st.sidebar.file_uploader("Upload Thermoelectric Data (CSV)", type="csv")
        doi_file = st.sidebar.file_uploader("Upload DOI Data (CSV)", type="csv")

        if data_file and doi_file:
            # 데이터 로드
            thermoelectric_df = load_and_process_data(data_file)
            doi_df = load_doi_data(doi_file)

        if thermoelectric_df is not None and doi_df is not None:
            # Thermoelectric 데이터와 DOI 데이터 병합
            merged_df = merge_thermoelectric_with_doi(thermoelectric_df, doi_df)
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

            # 공통 샘플 ID만 사이드바에 표시
            sample_id = st.sidebar.selectbox("Select Sample ID (with all properties):", sorted(common_samples))

            if sample_id:
               # 선택한 샘플 ID 데이터 필터링
                sample_data = filtered_df[filtered_df['sample_id'] == sample_id]
                st.write(f"### Data Table for Sample ID: {sample_id}")

                # 그래프 그리기
                st.write("### Property Graphs")
                dataframes = plot_TEP(filtered_df, sample_id)  # 데이터프레임 반환
                
                # DOI 정보 출력
                doi_info = sample_data[['DOI', 'URL']].drop_duplicates()
                if not doi_info.empty:
                    doi = doi_info['DOI'].iloc[0]
                    url = doi_info['URL'].iloc[0]

                    # DOI를 링크로 표시
                    st.write(f"**DOI**: {doi}")

                    # URL을 클릭 가능한 링크로 표시
                    st.markdown(f"**URL**: [{url}]({url})")
                else:
                    st.write("**DOI**: Not Available")
                    st.write("**URL**: Not Available")

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
        st.info("Please upload both data and DOI CSV files to proceed.")

if __name__ == "__main__":
    main()
