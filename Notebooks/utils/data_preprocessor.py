"""
데이터 전처리 검토 및 권장사항 제공
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt


def check_preprocessing_needs(df: pd.DataFrame, target_col: str = None) -> Dict:
    """
    데이터 전처리 필요사항 검토
    
    Parameters
    ----------
    df : DataFrame
        검토할 데이터프레임
    target_col : str, optional
        타겟 변수명 (제공되면 타겟과의 관계도 분석)
    
    Returns
    -------
    dict
        전처리 권장사항 딕셔너리
    """
    recommendations = {
        'weight_variables': [],
        'categorical_to_convert': [],
        'high_cardinality_categorical': [],
        'low_variance_features': [],
        'missing_values': {},
        'outliers': {},
        'data_leakage_risk': [],
        'recommendations': []
    }
    
    print("=" * 70)
    print("📊 데이터 전처리 검토 리포트")
    print("=" * 70)
    
    # 1. 가중치 변수 확인
    print("\n1️⃣ 가중치 변수 확인")
    print("-" * 70)
    weight_cols = [col for col in df.columns if 'wt' in col.lower() or 'weight' in col.lower()]
    if weight_cols:
        recommendations['weight_variables'] = weight_cols
        print(f"⚠️  발견된 가중치 변수: {weight_cols}")
        print("   → 가중치는 일반적으로 모델 학습 시 sample_weight로 사용하거나 제거합니다.")
        print("   → 특성으로 포함하면 모델 성능에 부정적 영향을 줄 수 있습니다.")
        recommendations['recommendations'].append({
            'type': 'weight',
            'action': '제거 또는 sample_weight로 사용',
            'columns': weight_cols
        })
    else:
        print("✅ 가중치 변수 없음")
    
    # 2. 범주형 변수 타입 확인
    print("\n2️⃣ 범주형 변수 타입 확인")
    print("-" * 70)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"범주형 변수 (object): {len(categorical_cols)}개")
    if categorical_cols:
        for col in categorical_cols:
            n_unique = df[col].nunique()
            print(f"  - {col}: {n_unique}개 고유값")
            if n_unique > 20:
                recommendations['high_cardinality_categorical'].append({
                    'column': col,
                    'n_unique': n_unique
                })
                print(f"    ⚠️  고유값이 많음 ({n_unique}개) - OneHotEncoding 시 차원 증가 주의")
    
    # 3. 숫자형이지만 범주형일 가능성이 있는 변수 확인
    print(f"\n숫자형 변수 중 범주형일 가능성 있는 변수:")
    potential_categorical = []
    for col in numeric_cols:
        n_unique = df[col].nunique()
        if n_unique <= 10 and n_unique < len(df) * 0.1:  # 고유값이 10개 이하이고 전체의 10% 미만
            potential_categorical.append({
                'column': col,
                'n_unique': n_unique,
                'values': sorted(df[col].unique().tolist())
            })
            print(f"  - {col}: {n_unique}개 고유값 {sorted(df[col].unique().tolist())}")
    
    if potential_categorical:
        recommendations['categorical_to_convert'] = potential_categorical
        recommendations['recommendations'].append({
            'type': 'categorical_conversion',
            'action': '범주형으로 변환 고려',
            'columns': [d['column'] for d in potential_categorical]
        })
    
    # 4. 결측치 확인
    print("\n3️⃣ 결측치 확인")
    print("-" * 70)
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        recommendations['missing_values'] = missing_cols.to_dict()
        print("⚠️  결측치가 있는 변수:")
        for col, count in missing_cols.items():
            pct = count / len(df) * 100
            print(f"  - {col}: {count}개 ({pct:.2f}%)")
            recommendations['recommendations'].append({
                'type': 'missing',
                'action': '결측치 처리 필요',
                'column': col,
                'count': count,
                'percentage': pct
            })
    else:
        print("✅ 결측치 없음")
    
    # 5. 분산이 낮은 변수 확인 (거의 모든 값이 동일한 경우)
    print("\n4️⃣ 분산이 낮은 변수 확인")
    print("-" * 70)
    low_variance = []
    for col in numeric_cols:
        if df[col].nunique() == 1:
            low_variance.append(col)
            print(f"  ⚠️  {col}: 모든 값이 동일 (제거 권장)")
        elif df[col].nunique() == 2:
            # 이진 변수인 경우, 한 클래스가 95% 이상이면 낮은 분산으로 간주
            value_counts = df[col].value_counts(normalize=True)
            if value_counts.max() >= 0.95:
                low_variance.append(col)
                print(f"  ⚠️  {col}: 한 값이 {value_counts.max()*100:.1f}% 차지 (제거 고려)")
    
    recommendations['low_variance_features'] = low_variance
    if low_variance:
        recommendations['recommendations'].append({
            'type': 'low_variance',
            'action': '제거 고려',
            'columns': low_variance
        })
    else:
        print("✅ 분산이 낮은 변수 없음")
    
    # 6. 데이터 누수 위험 변수 확인 (타겟 변수가 있는 경우)
    if target_col and target_col in df.columns:
        print("\n5️⃣ 데이터 누수(Data Leakage) 위험 변수 확인")
        print("-" * 70)
        y = df[target_col]
        leakage_risk = []
        
        for col in numeric_cols:
            if col == target_col:
                continue
            # 일치율 확인
            if df[col].dtype in [np.int64, np.int32]:
                match_rate = (df[col] == y).mean()
                if match_rate >= 0.95:
                    leakage_risk.append({
                        'column': col,
                        'match_rate': match_rate,
                        'reason': '타겟과 95% 이상 일치'
                    })
                    print(f"  ⚠️  {col}: 타겟과 {match_rate*100:.1f}% 일치")
            
            # 상관관계 확인
            corr = abs(df[col].corr(y))
            if corr >= 0.9:
                if col not in [r['column'] for r in leakage_risk]:
                    leakage_risk.append({
                        'column': col,
                        'correlation': corr,
                        'reason': '타겟과 상관관계 0.9 이상'
                    })
                    print(f"  ⚠️  {col}: 타겟과 상관관계 {corr:.4f}")
        
        recommendations['data_leakage_risk'] = leakage_risk
        if leakage_risk:
            recommendations['recommendations'].append({
                'type': 'data_leakage',
                'action': '제거 필수',
                'columns': [r['column'] for r in leakage_risk]
            })
        else:
            print("✅ 데이터 누수 위험 변수 없음")
    
    # 7. 이상치 확인 (IQR 방법)
    print("\n6️⃣ 이상치 확인 (IQR 방법)")
    print("-" * 70)
    outlier_summary = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:  # IQR이 0이 아닌 경우만
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                outlier_pct = outliers / len(df) * 100
                outlier_summary[col] = {
                    'count': outliers,
                    'percentage': outlier_pct,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
                if outlier_pct > 5:  # 이상치가 5% 이상인 경우만 경고
                    print(f"  ⚠️  {col}: {outliers}개 ({outlier_pct:.2f}%) 이상치")
    
    recommendations['outliers'] = outlier_summary
    if not outlier_summary:
        print("✅ 이상치가 많은 변수 없음 (5% 기준)")
    
    # 8. 종합 권장사항
    print("\n" + "=" * 70)
    print("📋 종합 권장사항")
    print("=" * 70)
    
    if not recommendations['recommendations']:
        print("✅ 특별한 전처리가 필요하지 않습니다.")
        print("   ClassifierTrainer가 자동으로 범주형 변수를 OneHotEncoding하고,")
        print("   연속형 변수를 StandardScaler로 스케일링합니다.")
    else:
        for i, rec in enumerate(recommendations['recommendations'], 1):
            print(f"\n{i}. {rec['type'].upper()}: {rec['action']}")
            if 'columns' in rec:
                print(f"   대상 변수: {rec['columns']}")
            elif 'column' in rec:
                print(f"   대상 변수: {rec['column']}")
    
    print("\n" + "=" * 70)
    
    return recommendations


def preprocess_dataframe(
    df: pd.DataFrame,
    target_col: str = None,
    drop_weight: bool = True,
    convert_categorical: list = None,
    convert_ordinal: list = None,
    convert_binary: list = None,
    drop_low_variance: bool = False,
    drop_leakage: bool = True
) -> pd.DataFrame:
    """
    데이터프레임 전처리 실행

    Parameters
    ----------
    df : DataFrame
        전처리할 데이터프레임
    target_col : str, optional
        타겟 변수명
    drop_weight : bool
        가중치 변수 제거 여부 (default=True)
    convert_categorical : list, optional
        범주형으로 변환할 변수 리스트
    convert_ordinal : list, optional
        순서형으로 변환할 변수 리스트
    convert_binary : list, optional
        이진형(0/1)으로 변환할 변수 리스트
    drop_low_variance : bool
        분산이 낮은 변수 제거 여부 (default=False)
    drop_leakage : bool
        데이터 누수 위험 변수 제거 여부 (default=True)

    Returns
    -------
    DataFrame
        전처리된 데이터프레임
    """
    import pandas as pd

    df_processed = df.copy()

    # 1. 가중치 변수 제거
    if drop_weight:
        weight_cols = [col for col in df_processed.columns if 'wt' in col.lower() or 'weight' in col.lower()]
        if weight_cols:
            df_processed = df_processed.drop(columns=weight_cols)
            print(f"✅ 가중치 변수 제거: {weight_cols}")

    # 2. 범주형으로 변환
    if convert_categorical:
        for col in convert_categorical:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype('category')
                # print(f"✅ {col}을 범주형으로 변환")

    # 2-1. 순서형으로 변환
    if convert_ordinal:
        for col in convert_ordinal:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype('category')
                df_processed[col] = df_processed[col].cat.as_ordered()
                # print(f"✅ {col}을 순서형(category, ordered)으로 변환")

    # 2-2. 이진형으로 변환 (0과 1로 매핑)
    if convert_binary:
        for col in convert_binary:
            if col in df_processed.columns:
                unique_vals = sorted(df_processed[col].dropna().unique())
                if len(unique_vals) == 2:
                    bin_map = {unique_vals[0]: 0, unique_vals[1]: 1}
                    df_processed[col] = df_processed[col].map(bin_map)
                else:
                    # 이미 0/1이 아닌 경우에만 에러 출력
                    print(f"⚠️  {col} 변수는 2개의 값이 아닙니다: {unique_vals}")
                # print(f"✅ {col}을 이진형(0/1)으로 변환")

    # 3. 분산이 낮은 변수 제거
    if drop_low_variance:
        import numpy as np
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        low_variance_cols = []
        for col in numeric_cols:
            if col == target_col:
                continue
            if df_processed[col].nunique() == 1:
                low_variance_cols.append(col)
            elif df_processed[col].nunique() == 2:
                value_counts = df_processed[col].value_counts(normalize=True)
                if value_counts.max() >= 0.95:
                    low_variance_cols.append(col)

        if low_variance_cols:
            df_processed = df_processed.drop(columns=low_variance_cols)
            print(f"✅ 분산이 낮은 변수 제거: {low_variance_cols}")

    # 4. 데이터 누수 위험 변수 제거
    if drop_leakage and target_col and target_col in df_processed.columns:
        recommendations = check_preprocessing_needs(df_processed, target_col)
        leakage_cols = [r['column'] for r in recommendations['data_leakage_risk']]
        if leakage_cols:
            df_processed = df_processed.drop(columns=leakage_cols)
            print(f"✅ 데이터 누수 위험 변수 제거: {leakage_cols}")

    return df_processed



import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

def categorical_preprocess(df):
    """
    순서가 없는 범주형 (category dtype, ordered=False) 처리: 
    - 결측치처리 : 9999로 대체 
    - OneHotEncoding
    """
    df = df.copy()
    # category dtype, ordered=False 만 선택
    cat_cols = [c for c in df.select_dtypes(['category']).columns
                if not df[c].cat.ordered]
    # 결측값을 9999로 대체
    for col in cat_cols:
        # category dtype에서는 새로운 카테고리를 먼저 추가해야 함
        if '9999' not in df[col].cat.categories:
            df[col] = df[col].cat.add_categories('9999')
        df[col] = df[col].fillna('9999')
        # OneHotEncoder를 위해 모든 값을 문자열로 변환 (타입 통일)
        df[col] = df[col].astype(str)
    if cat_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = encoder.fit_transform(df[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
        # 기존 categorical 컬럼 제거 후 결합
        df = df.drop(columns=cat_cols)
        df = pd.concat([df, encoded_df], axis=1)
    return df

def ordinal_categorical_preprocess(df):
    """
    순서가 있는 category 처리: category dtype, ordered=True
    - NaN처리 : Median값
    - pandas의 category 코드값(int)로 변환
    """
    df = df.copy()
    ord_cols = [c for c in df.select_dtypes(['category']).columns
                if df[c].cat.ordered]
    for col in ord_cols:
        # 카테고리 코드값(int)로 변환 (읽기 전용이므로 복사본 생성)
        codes = df[col].cat.codes.copy()
        # 결측치를 가진 인덱스 찾기 (cat.codes에서 결측치는 -1)
        nan_idx = codes[codes == -1].index
        # 결측치가 있을 때 median code로 대체
        if len(nan_idx) > 0:
            valid_codes = codes[codes != -1]
            if len(valid_codes) > 0:
                median_code = int(np.median(valid_codes))
                codes.loc[nan_idx] = median_code
            else:
                # 모든 값이 결측치인 경우 0으로 대체
                codes.loc[nan_idx] = 0
        df[col] = codes
    return df

def object_preprocess(df):
    """
    object dtype 처리: 
    - NaN을 "Unknown"으로 대체(이상치값) 후, OrdinalEncoder
    """
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object"]).columns
    if len(obj_cols) == 0:
        return df
    df[obj_cols] = df[obj_cols].fillna("Unknown")
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df[obj_cols] = encoder.fit_transform(df[obj_cols])
    return df

def integer_preprocess(df):
    """ 
    int dtype: NaN은 median으로 대체 후 Z-표준화(StandardScaler) 적용
    """
    df = df.copy()
    int_cols = df.select_dtypes(include=["int", "int64"]).columns
    if len(int_cols):
        median_vals = df[int_cols].median()
        df[int_cols] = df[int_cols].fillna(median_vals)
        scaler = StandardScaler()
        df[int_cols] = scaler.fit_transform(df[int_cols])
    return df

def float_preprocess(df):
    """
    float dtype: NaN은 median으로 대체 + 표준화(StandardScaler)
    """
    df = df.copy()
    float_cols = df.select_dtypes(include=["float", "float64"]).columns
    if len(float_cols):
        median_vals = df[float_cols].median()
        df[float_cols] = df[float_cols].fillna(median_vals)
        scaler = StandardScaler()
        df[float_cols] = scaler.fit_transform(df[float_cols])
    return df

def data_preprocess_pipeline(df):
    """
    dtype별 전처리 통합 파이프라인: 
      1. 순서 있는 category (ordinal) → ordinal_categorical_preprocess (NaN은 Median)
      2. object → object_preprocess (NaN은 Unknown)
      3. 순서 없는 category → categorical_preprocess (NaN은 Unknown)
      4. int → integer_preprocess (NaN은 Median)
      5. float → float_preprocess (NaN은 Median)
    """

    print("▶ integer 전처리 중...")
    df = integer_preprocess(df)

    print("▶ float 전처리 중...")
    df = float_preprocess(df)
    
    print("▶ 순서 있는 category(ordinal) 전처리 중...")
    df = ordinal_categorical_preprocess(df)

    print("▶ object 전처리 중...")
    df = object_preprocess(df)

    print("▶ 순서 없는 category 전처리 중...")
    df = categorical_preprocess(df)


    print("✅ 데이터 전처리 완료")
    return df
