import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def stratified_train_test_split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Stratified Train/Test Split for imbalanced datasets.
    클래스 비율을 유지하면서 8:2로 Train/Test를 분리합니다.

    Parameters
    ----------
    df : DataFrame
        전체 데이터셋
    target_col : str
        라벨 컬럼명
    test_size : float
        테스트 데이터 비율 (default=0.2)
    random_state : int
        랜덤 시드 (재현성 확보용)

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple
        Stratified하게 분리된 학습/테스트 데이터셋
    """

    X = df.drop(columns=[target_col])
    y = df[target_col]

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )

    for train_idx, test_idx in sss.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # 분포 확인용 로그
    def ratio(y):
        counts = y.value_counts()
        return {cls: round(cnt / len(y) * 100, 2) for cls, cnt in counts.items()}

    print("============================================================")
    print(f"전체 데이터 크기: {len(df)}")
    print(f"Train: {len(X_train)} samples, Class 분포: {ratio(y_train)}")
    print(f"Test : {len(X_test)} samples,  Class 분포: {ratio(y_test)}")
    print("============================================================")
    print(f"Train 클래스 분포: {y_train.value_counts().to_dict()}")
    print(f"Test 클래스 분포: {y_test.value_counts().to_dict()}")
    print("============================================================")
    print(f"Train 불균형 비율: {y_train.value_counts()[0] / y_train.value_counts()[1]:.2f}:1")
    print(f"Test 불균형 비율: {y_test.value_counts()[0] / y_test.value_counts()[1]:.2f}:1")
    print("============================================================")

    # return pd.concat([X_train, y_train], axis=1), pd.concat([X_test, y_test], axis=1)
    return X_train, X_test, y_train, y_test


def balanced_random_sampling_train_test_split(df, target_col, test_size=0.2, random_state=42):
    """
    Train / Test를 8:2로 나누는데, test set에서 각 클래스가 동일한 수로 담기도록 샘플링한다.
    (예: test에 class 1이 35개 들어가면 class 0도 35개만 들어감)
    나머지 데이터는 모두 train으로 사용함. 
    """
    import numpy as np
    np.random.seed(random_state)

    # Class별로 인덱스 분리
    idx_class_0 = df[df[target_col] == 0].index.tolist()
    idx_class_1 = df[df[target_col] == 1].index.tolist()

    # test set에서 가질 클래스별 최대 개수
    n_test_class_1 = int(len(idx_class_1) * test_size)
    n_test_class_0 = int(len(idx_class_0) * test_size)
    n_test_per_class = min(n_test_class_0, n_test_class_1)
    # 혹시라도 너무 작아지는 것 방지 (최소 1개)
    n_test_per_class = max(1, n_test_per_class)

    # 랜덤 샘플링
    test_idx_1 = np.random.choice(idx_class_1, n_test_per_class, replace=False)
    test_idx_0 = np.random.choice(idx_class_0, n_test_per_class, replace=False)

    test_idx = np.concatenate([test_idx_0, test_idx_1])
    train_idx = df.index.difference(test_idx)

    df_train = df.loc[train_idx].reset_index(drop=True)
    df_test = df.loc[test_idx].reset_index(drop=True)

    X_train = df_train.drop(target_col, axis=1)
    y_train = df_train[target_col]
    X_test = df_test.drop(target_col, axis=1)
    y_test = df_test[target_col]

    print(f"Train set 클래스 분포: {y_train.value_counts()}")
    print(f"Test set 클래스 분포: {y_test.value_counts()}")
    return X_train, X_test, y_train, y_test

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN
import numpy as np
import pandas as pd 

def oversample_train_test_split(
    X, 
    y, 
    target_col='ODD_CP',
    test_size_per_class=60,
    train_size_per_class=240,
    random_state=42,
    verbose=True,
    method='SMOTE'
):
    """
    언더샘플링(클래스 0) 및 오버샘플링(클래스 1)을 적용한 Train/Test 분할
    
    Parameters
    ----------
    X : DataFrame
        입력 feature
    y : Series or array
        타겟 변수 (0 또는 1)
    target_col : str, optional
        타겟 변수 컬럼명 (X와 y가 DataFrame인 경우 사용)
    test_size_per_class : int
        Test set에 사용할 각 클래스의 샘플 수 (default=60)
    train_size_per_class : int
        Train set에 사용할 각 클래스의 샘플 수 (default=240)
    random_state : int
        랜덤 시드 (default=42)
    verbose : bool
        진행 상황 출력 여부 (default=True)
    method : str
        샘플링 방법 (default='SMOTE')
        'SMOTE' : SMOTE (기본값)
        'ADASYN' : ADASYN
        'SMOTETomek' 또는 'SMOTETOMEK' : SMOTETomek
        'SMOTEENN' 또는 'SMOTEEN' : SMOTEENN
        대소문자 구분 없음, 오타 자동 수정됨
    Returns
    -------
    X_train, X_test, y_train, y_test : tuple
        샘플링된 Train/Test 데이터셋
    """
    # DataFrame으로 변환
    if isinstance(X, pd.DataFrame):
        df = pd.concat([X, y], axis=1)
    else:
        df = pd.DataFrame(X)
        df[target_col] = y
    
    if verbose:
        print("="*60)
        print("데이터 분할 및 샘플링")
        print("="*60)
    
    # 전체 데이터에서 클래스별 분리
    df_class_1 = df[df[target_col] == 1].copy()
    df_class_0 = df[df[target_col] == 0].copy()
    
    if verbose:
        print(f"\n전체 데이터:")
        print(f"  클래스 0: {len(df_class_0)}개")
        print(f"  클래스 1: {len(df_class_1)}개")
    
    # 1단계: Test set 구성 (각 클래스에서 원본 지정 개수씩)
    if verbose:
        print(f"\n1단계: Test set 구성 (각 클래스 {test_size_per_class}개씩)")
    
    # 각 클래스에서 test_size_per_class개씩 샘플링
    n_test_0 = min(test_size_per_class, len(df_class_0))
    n_test_1 = min(test_size_per_class, len(df_class_1))
    
    df_0_test = df_class_0.sample(n=n_test_0, random_state=random_state)
    df_1_test = df_class_1.sample(n=n_test_1, random_state=random_state)
    df_test = pd.concat([df_0_test, df_1_test], axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]
    
    if verbose:
        print(f"  Test set: 클래스 0={sum(y_test==0)}개, 클래스 1={sum(y_test==1)}개 (총 {len(y_test)}개)")
    
    # 2단계: Train set용 원본 데이터 (Test set 제외)
    if verbose:
        print(f"\n2단계: Train set용 원본 데이터 (Test set 제외)")
    
    df_0_train_orig = df_class_0.drop(df_0_test.index)
    df_1_train_orig = df_class_1.drop(df_1_test.index)
    
    if verbose:
        print(f"  Train 원본: 클래스 0={len(df_0_train_orig)}개, 클래스 1={len(df_1_train_orig)}개")
    
    # 3단계: Train set 샘플링
    if verbose:
        print(f"\n3단계: Train set 샘플링")
    
    # SMOTE는 두 클래스가 모두 필요하므로, 클래스 0과 1을 함께 사용
    if len(df_1_train_orig) < train_size_per_class:
        # 3-1. 클래스 0과 1을 합쳐서 SMOTE 적용 (클래스 1만 오버샘플링)
        df_train_temp = pd.concat([df_0_train_orig, df_1_train_orig], axis=0)
        X_train_temp = df_train_temp.drop(columns=[target_col])
        y_train_temp = df_train_temp[target_col]
        
        # method 정규화 (대소문자 무시, 오타 자동 수정)
        method_upper = method.upper()
        if 'SMOTEEN' in method_upper :
            method_upper = 'SMOTEENN'  # SMOTEEN -> SMOTEENN 자동 수정
        elif 'SMOTETOMEK' in method_upper:
            method_upper = 'SMOTETOMEK'
        elif 'ADASYN' in method_upper:
            method_upper = 'ADASYN'
        elif 'SMOTE' in method_upper:
            method_upper = 'SMOTE'
        
        # 오버샘플링 방법으로 클래스 1을 train_size_per_class개로 오버샘플링 (클래스 0은 그대로 유지)
        if method_upper == 'SMOTE':
            sampler = SMOTE(
                sampling_strategy={0: len(df_0_train_orig), 1: train_size_per_class},
                random_state=random_state, 
                k_neighbors=min(5, len(df_1_train_orig)-1)
            )
            method_name = 'SMOTE'
        elif method_upper == 'ADASYN':
            sampler = ADASYN(
                sampling_strategy={0: len(df_0_train_orig), 1: train_size_per_class},
                random_state=random_state,
                n_neighbors=min(5, len(df_1_train_orig)-1)
            )
            method_name = 'ADASYN'
        elif method_upper == 'SMOTETOMEK':
            sampler = SMOTETomek(
                sampling_strategy={0: len(df_0_train_orig), 1: train_size_per_class},
                random_state=random_state
            )
            method_name = 'SMOTETomek'
        elif method_upper == 'SMOTEENN':
            sampler = SMOTEENN(
                sampling_strategy={0: len(df_0_train_orig), 1: train_size_per_class},
                random_state=random_state
            )
            method_name = 'SMOTEENN'
        else:
            # 기본값으로 SMOTE 사용
            if verbose:
                print(f"  경고: 알 수 없는 method '{method}'. 기본값 'SMOTE'를 사용합니다.")
            sampler = SMOTE(
                sampling_strategy={0: len(df_0_train_orig), 1: train_size_per_class},
                random_state=random_state, 
                k_neighbors=min(5, len(df_1_train_orig)-1)
            )
            method_name = 'SMOTE'
        
        X_train_sampled, y_train_sampled = sampler.fit_resample(X_train_temp, y_train_temp)
        
        # 샘플링된 데이터 분리
        df_train_sampled = pd.concat([X_train_sampled, pd.Series(y_train_sampled, name=target_col)], axis=1)
        df_0_train_sampled = df_train_sampled[df_train_sampled[target_col] == 0]
        df_1_train = df_train_sampled[df_train_sampled[target_col] == 1].reset_index(drop=True)
        
        # 3-2. 클래스 0: 언더샘플링 (train_size_per_class개)
        n_train_0 = min(train_size_per_class, len(df_0_train_sampled))
        df_0_train = df_0_train_sampled.sample(n=n_train_0, random_state=random_state)
        
        if verbose:
            print(f"  클래스 0 언더샘플링: {len(df_0_train_orig)}개 → {len(df_0_train)}개")
            print(f"  클래스 1 오버샘플링: {len(df_1_train_orig)}개 → {len(df_1_train)}개 ({method_name} 사용)")
    else:
        # 이미 충분하면 그대로 사용
        n_train_0 = min(train_size_per_class, len(df_0_train_orig))
        n_train_1 = min(train_size_per_class, len(df_1_train_orig))
        df_0_train = df_0_train_orig.sample(n=n_train_0, random_state=random_state)
        df_1_train = df_1_train_orig.sample(n=n_train_1, random_state=random_state)
        if verbose:
            print(f"  클래스 0 언더샘플링: {len(df_0_train_orig)}개 → {len(df_0_train)}개")
            print(f"  클래스 1 언더샘플링: {len(df_1_train_orig)}개 → {len(df_1_train)}개")
    
    # 4단계: Train set 최종 구성
    if verbose:
        print(f"\n4단계: Train set 최종 구성")
    
    df_train = pd.concat([df_0_train, df_1_train], axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    
    if verbose:
        print(f"  Train set: 클래스 0={sum(y_train==0)}개, 클래스 1={sum(y_train==1)}개 (총 {len(y_train)}개)")
        if sum(y_train==1) > 0:
            print(f"  Train 비율: 1:{sum(y_train==0)/sum(y_train==1):.1f}")
        
        print(f"\n최종 데이터 분포:")
        print(f"  Train: {len(X_train)}개 (클래스 0={sum(y_train==0)}, 클래스 1={sum(y_train==1)})")
        print(f"  Test:  {len(X_test)}개 (클래스 0={sum(y_test==0)}, 클래스 1={sum(y_test==1)})")
        print("="*60)
    
    return X_train, X_test, y_train, y_test


def downsample_train_test_split(df, target_col, n_train_class0=None, n_test_per_class=60, random_state=42, verbose=True):
    """
    Test set: 클래스 1 60개, 클래스 0 60개
    Train set: 
        - 클래스 0: n_train_class0 지정 (None이면 test set 제외 모두)
        - 클래스 1: 남은 전부 사용
    """
    df_class_1 = df[df[target_col] == 1].copy()
    df_class_0 = df[df[target_col] == 0].copy()

    if verbose:
        print(f"\n전체 데이터:")
        print(f"  클래스 0: {len(df_class_0)}개")
        print(f"  클래스 1: {len(df_class_1)}개")
    
    # 1단계: Test set 구성 (각 클래스에서 n_test_per_class개씩)
    if verbose:
        print(f"\n1단계: Test set 구성 (각 클래스 {n_test_per_class}개씩)")

    n_test_0 = min(n_test_per_class, len(df_class_0))
    n_test_1 = min(n_test_per_class, len(df_class_1))

    df_0_test = df_class_0.sample(n=n_test_0, random_state=random_state)
    df_1_test = df_class_1.sample(n=n_test_1, random_state=random_state)
    df_test = pd.concat([df_0_test, df_1_test], axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)

    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    # 2단계: 남은 데이터에서 train set 구성
    df_0_remain = df_class_0.drop(df_0_test.index)
    df_1_remain = df_class_1.drop(df_1_test.index)

    # n_train_class0이 None이면 test set 제외 전부 사용
    if n_train_class0 is None:
        n_train_0 = len(df_0_remain)
    else:
        n_train_0 = min(n_train_class0, len(df_0_remain))

    n_train_1 = len(df_1_remain)

    if n_train_0 > 0:
        df_0_train = df_0_remain.sample(n=n_train_0, random_state=random_state)
    else:
        df_0_train = pd.DataFrame(columns=df_0_remain.columns)

    df_1_train = df_1_remain  # 남은 전부 사용

    df_train = pd.concat([df_0_train, df_1_train], axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    if verbose:
        print(f"Train set 클래스 0 (샘플링): {len(df_0_train)}개")
        print(f"Train set 클래스 1 (남은 전부): {len(df_1_train)}개")
        print(f"Test set 클래스 0: {len(df_0_test)}개")
        print(f"Test set 클래스 1: {len(df_1_test)}개")

    return X_train, X_test, y_train, y_test
