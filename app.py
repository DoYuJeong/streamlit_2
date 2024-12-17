import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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

# Streamlit 메인 함수
def main():
    st.title("Thermoelectric Property Viewer")

    # 데이터 로드
    file_paths = {
        'sigma': 'df_combined_sigma.csv',
        'alpha': 'df_alpha0.csv',
        'kappa': 'df_kappa0.csv',
        'ZT': 'df_combined_ZT.csv'
    }

    # 데이터프레임 딕셔너리 구성
    dataframes = {}
    for key, path in file_paths.items():
        try:
            dataframes[key] = pd.read_csv(path)
        except FileNotFoundError:
            st.warning(f"File {path} not found.")
    
    # 공통 sample_id 추출
    common_sample_ids = get_common_sample_ids(dataframes)

    if not common_sample_ids:
        st.error("No common sample IDs found across all datasets.")
        return

    # 사용자 선택: sample_id
    selected_sample_id = st.sidebar.selectbox("Select Sample ID:", sorted(common_sample_ids))

    # 데이터프레임 출력
    st.write(f"### Selected Sample ID: {selected_sample_id}")

    # 정확한 데이터프레임 출력
    if 'sigma' in dataframes and not dataframes['sigma'].empty:
        df_sigma_filtered = dataframes['sigma'][dataframes['sigma']['sample_id'] == selected_sample_id]
        st.write("#### Sigma DataFrame")
        st.dataframe(df_sigma_filtered)

    if 'ZT' in dataframes and not dataframes['ZT'].empty:
        df_ZT_filtered = dataframes['ZT'][dataframes['ZT']['sample_id'] == selected_sample_id]
        st.write("#### ZT DataFrame")
        st.dataframe(df_ZT_filtered)

    if 'alpha' in dataframes and not dataframes['alpha'].empty:
        df_alpha_filtered = dataframes['alpha'][dataframes['alpha']['sample_id'] == selected_sample_id]
        st.write("#### Alpha DataFrame")
        st.dataframe(df_alpha_filtered)

    if 'kappa' in dataframes and not dataframes['kappa'].empty:
        df_kappa_filtered = dataframes['kappa'][dataframes['kappa']['sample_id'] == selected_sample_id]
        st.write("#### Kappa DataFrame")
        st.dataframe(df_kappa_filtered)

    # 그래프 출력
    st.write("### Graphs for Selected Sample ID")
    create_and_plot_graphs_filtered(dataframes, selected_sample_id)

if __name__ == "__main__":
    main()
