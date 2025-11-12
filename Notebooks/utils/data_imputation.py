import pandas as pd
import matplotlib.pyplot as plt

def filter_by_missing_ratio(df, threshold=0.2, visualize=True):
    """
    결측치 비율 기준(threshold)으로 컬럼을 필터링하고, 결과 및 시각화를 반환합니다.
    - threshold: 남기고 싶은 컬럼의 최대 결측치 허용비율(0~1), default=0.05
    - visualize: 결측치 남은 컬럼의 비율 시각화 여부
    """
    # ' ' 또는 공백 값을 NaN (결측치)로 변환
    df = df.replace(r'^\s*$', pd.NA, regex=True)

    # 결측치 비율 구하기
    missing_ratio = df.isnull().mean()

    # threshold 미만인 컬럼만 남기기
    cols_under_thresh_na = missing_ratio[missing_ratio < threshold].index
    deleted_cols = [col for col in df.columns if col not in cols_under_thresh_na]
    filtered_df = df[cols_under_thresh_na]
    print(f"삭제 칼럼들 : {deleted_cols}")

    print(f"\n==== 결측치가 {int(threshold*100)}% 미만인 컬럼만 남김 ====")
    print(f"남은 변수 수: {len(filtered_df.columns)}")
    print("남은 컬럼 리스트:", list(filtered_df.columns))

    # 결측치가 남아있는(남긴) 컬럼만 추출해서 결측치 개수 출력
    still_missing_cols = filtered_df.columns[filtered_df.isnull().any()]
    print(f"\n==== 남아있는 컬럼 중 결측치가 있는 컬럼의 결측치 개수 ====")
    print(filtered_df[still_missing_cols].isnull().sum())
    print(f"\n결측치가 남아있는 변수 수: {len(still_missing_cols)} / 전체 변수 수: {len(filtered_df.columns)}")

    # threshold 미만 컬럼 중 결측치 비율 시각화
    if visualize and len(still_missing_cols) > 0:
        still_missing_ratio = filtered_df[still_missing_cols].isnull().mean().sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        still_missing_ratio.plot(kind="bar")
        plt.title(f"Missing Value Ratio (%) (<{int(threshold*100)}% NA Columns Only)")
        plt.ylabel("Missing Value Ratio")
        plt.xlabel("Column Name")
        plt.tight_layout()
        plt.show()
        
    return filtered_df