import pandas as pd 
import warnings
import numpy as np
import pickle
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================================================
# 1. 데이터 로드 및 전처리
# ============================================================================
print("="*80)
print("데이터 로드 및 전처리")
print("="*80)

# train_df와 test_df 로드
train_df = pd.read_csv("/workspace/data/tune_data/tune_df.csv")
test_df = pd.read_csv("/workspace/data/tune_data/test_df.csv")

print(f"Train 데이터 크기: {len(train_df)}")
print(f"Test 데이터 크기: {len(test_df)}")

# y = 타겟 변수 / x = 예측 변수
except_cols = ['Z_KDBDRS', 'K_ODD', 'KDBDRS', 'wt_s', 'DICCD_CP', 'ODD_Cur', 'ODD_Pas', 'DICCD_C', 'DICCD_P', 'MentD']
standard_cols = ['ZAI_Incom', 'Z_K_ODD', 'Z_K_CD', 'Z_K_IA', 'Z_K_HI', 'Z_GAD', 'Z_PHQ', 'Z_SAS']
nonstandard_cols = ['Incom', 'K_ODD', 'K_CD', 'K_IA', 'K_HI', 'GAD', 'PHQ', 'SAS']

drop_cols = except_cols + nonstandard_cols

# Train 데이터 전처리
print(f"\nTrain 데이터 - 기존 변수 수: {len(train_df.columns)}")
train_df = train_df.drop(columns=[col for col in drop_cols if col in train_df.columns])
print(f"Train 데이터 - 남은 변수 수: {len(train_df.columns)}")

# Test 데이터 전처리
print(f"\nTest 데이터 - 기존 변수 수: {len(test_df.columns)}")
test_df = test_df.drop(columns=[col for col in drop_cols if col in test_df.columns])
print(f"Test 데이터 - 남은 변수 수: {len(test_df.columns)}")

from Notebooks.utils.data_imputation import filter_by_missing_ratio

# Train 데이터 결측치 처리
train_df = filter_by_missing_ratio(train_df, threshold=0.25, visualize=False)
print(f"Train 데이터 결측치 처리 후 크기: {len(train_df)}")

# Test 데이터 결측치 처리
test_df = filter_by_missing_ratio(test_df, threshold=0.25, visualize=False)
print(f"Test 데이터 결측치 처리 후 크기: {len(test_df)}")

from Notebooks.utils.data_preprocessor import check_preprocessing_needs, preprocess_dataframe
from Notebooks.utils.data_preprocessor import data_preprocess_pipeline

# Train 데이터 전처리
X_train_raw = train_df.drop(columns=['ODD_CP'])
y_train_raw = train_df['ODD_CP']

X_train_raw = preprocess_dataframe(
    X_train_raw, 
    target_col='ODD_CP',
    drop_weight=True,
    convert_categorical=['Answ', 'IGD_P', 'FEdu', 'MEdu', 'FJob', 'MJob', 'Age_Grp', 'P_Marr'],
    convert_ordinal=['ST1', 'ST2', 'ST3', 'ST4', 'PAF', 'MAlc', 'FAlc', "MTob", "FTob", "MAlc", "FAlc", "GAlc", "MTob", "FTob", "GTob"], 
    convert_binary=['SRD_CP', 'IGD_P', 'Sex', 'PSleep', 'SBV', 'SBP', 'CBV', 'CBP', 'GDec', 'BF', 'RFG', 'MentD', 'AdolSlp', 'MoodD', 'AnxD'],
    drop_low_variance=False,
    drop_leakage=True
)
X_train_raw = data_preprocess_pipeline(X_train_raw)

# Test 데이터 전처리 (동일한 전처리 적용)
X_test_raw = test_df.drop(columns=['ODD_CP'])
y_test_raw = test_df['ODD_CP']

X_test_raw = preprocess_dataframe(
    X_test_raw, 
    target_col='ODD_CP',
    drop_weight=True,
    convert_categorical=['Answ', 'IGD_P', 'FEdu', 'MEdu', 'FJob', 'MJob', 'Age_Grp', 'P_Marr'],
    convert_ordinal=['ST1', 'ST2', 'ST3', 'ST4', 'PAF', 'MAlc', 'FAlc', "MTob", "FTob", "MAlc", "FAlc", "GAlc", "MTob", "FTob", "GTob"], 
    convert_binary=['SRD_CP', 'IGD_P', 'Sex', 'PSleep', 'SBV', 'SBP', 'CBV', 'CBP', 'GDec', 'BF', 'RFG', 'MentD', 'AdolSlp', 'MoodD', 'AnxD'],
    drop_low_variance=False,
    drop_leakage=True
)
X_test_raw = data_preprocess_pipeline(X_test_raw)

# 컬럼 일치 확인 및 정렬
common_cols = list(set(X_train_raw.columns) & set(X_test_raw.columns))
X_train_raw = X_train_raw[common_cols]
X_test_raw = X_test_raw[common_cols]

print(f"\n전처리 완료!")
print(f"  Train: {len(X_train_raw)}개 샘플, {len(X_train_raw.columns)}개 변수")
print(f"  Test: {len(X_test_raw)}개 샘플, {len(X_test_raw.columns)}개 변수")
print(f"  Train 클래스 분포: {y_train_raw.value_counts().to_dict()}")
print(f"  Test 클래스 분포: {y_test_raw.value_counts().to_dict()}")

# K-Fold로 나눌 때 각 fold의 train 부분 크기 계산 (5-Fold 기준 약 80%)
# 전체 train: 3262개 -> 각 fold의 train 부분: 약 2,609개 (80%)
# 클래스 0: 3145개 -> 각 fold의 train 부분: 약 2,516개 (3145 * 0.8)
# 클래스 1: 117개 -> 각 fold의 train 부분: 약 94개 (117 * 0.8)
n_folds = 5
fold_train_ratio = (n_folds - 1) / n_folds  # 5-Fold: 0.8
train_class0_count = y_train_raw.value_counts().get(0, 0)
train_class1_count = y_train_raw.value_counts().get(1, 0)

# 각 fold의 train 부분에서 예상되는 클래스 0 크기 (전체 train의 클래스 0 크기 기준)
# 전체 train에서 test를 제외한 클래스 0 크기 = 3145개
# 각 fold의 train 부분: 약 2,516개 (3145 * 0.8)
fold_train_class0_size = int(train_class0_count * fold_train_ratio)

print(f"\nK-Fold 설정 (n_splits={n_folds}):")
print(f"  각 fold의 train 부분 비율: {fold_train_ratio:.1%}")
print(f"  각 fold의 train 부분 예상 크기: 약 {int(len(X_train_raw) * fold_train_ratio)}개")
print(f"  각 fold의 train 부분 클래스 0 예상 크기: 약 {fold_train_class0_size}개")
print(f"  각 fold의 train 부분 클래스 1 예상 크기: 약 {int(train_class1_count * fold_train_ratio)}개")

# ============================================================================
# 2. 샘플링 방법 정의
# ============================================================================
SAMPLING_METHODS = {
    # Test Set(60/60) -> Class 0(Major Class) : w&w/o Random Downsampling <-> SMOTE/SMOTEEN/SMOTETomek/ADASYN
    # baseline 
    'baseline': {
        'type': 'downsample',
        'params': {'n_train_class0': None}
    },
    # downsample
    'downsample': {
        'type': 'downsample',
        'params': {'n_train_class0': 240}
    },
    
    # Minor Class oversample
    # 각 fold의 train 부분 크기를 고려하여 조정 (전체 train의 클래스 0 크기 기준)
    'oversample_SMOTE': {
        'type': 'oversample',
        'params': {'train_size_per_class': fold_train_class0_size, 'method': 'SMOTE'}
    },
    'oversample_SMOTEEN': {
        'type': 'oversample',
        'params': {'train_size_per_class': fold_train_class0_size, 'method': 'SMOTEENN'}
    },
    'oversample_SMOTETomek': {
        'type': 'oversample',
        'params': {'train_size_per_class': fold_train_class0_size, 'method': 'SMOTETomek'}
    },
    'oversample_ADASYN': {
        'type': 'oversample',
        'params': {'train_size_per_class': fold_train_class0_size, 'method': 'ADASYN'}
    },
    
    # Major Class downsample + Minor Class oversample
    'downsample_SMOTE': {
        'type': 'oversample',
        'params': {'train_size_per_class': 240, 'method': 'SMOTE'}
    },
    'downsample_SMOTEEN': {
        'type': 'oversample',
        'params': {'train_size_per_class': 240, 'method': 'SMOTEENN'}
    },
    'downsample_SMOTETomek': {
        'type': 'oversample',
        'params': {'train_size_per_class': 240, 'method': 'SMOTETomek'}
    },
    'downsample_ADASYN': {
        'type': 'oversample',
        'params': {'train_size_per_class': 240, 'method': 'ADASYN'}
    },
}

# ============================================================================
# 3. 실험 실행
# ============================================================================
from Notebooks.utils.ml_model import MultiModelFoldTrainer

# 결과 저장 디렉토리 생성
results_dir = Path("/workspace/data/results")
results_dir.mkdir(parents=True, exist_ok=True)

# 각 샘플링 방법에 대해 실험 실행
for sampling_name, sampling_config in SAMPLING_METHODS.items():
    print(f"\n\n{'#'*80}")
    print(f"# 실험 {list(SAMPLING_METHODS.keys()).index(sampling_name) + 1}/{len(SAMPLING_METHODS)}: {sampling_name}")
    print(f"{'#'*80}\n")
    
    try:
        # MultiModelFoldTrainer에 샘플링 설정 전달
        multi_model_trainer = MultiModelFoldTrainer(
            models_to_train=None,  # 모든 가능한 모델 사용
            n_splits=5, 
            random_state=42, 
            T=0.01,
            sampling_config=sampling_config  # 샘플링 설정 전달
        )
        
        # 원본 train 데이터와 test 데이터 전달
        # MultiModelFoldTrainer 내부에서 K-Fold 분할 후 각 fold의 train 부분에만 oversampling 적용
        multi_model_trainer.fit(
            X_train_raw, y_train_raw, 
            X_test_raw, y_test=y_test_raw
        )
        
        # 3. 결과 저장
        print(f"\n{'='*80}")
        print(f"결과 저장 중: {sampling_name}")
        print(f"{'='*80}\n")
        
        try:
            save_dict = {
                'sampling_method': sampling_name,
                'sampling_config': sampling_config,
                'test_inputs': X_test_raw,
                'test_labels': multi_model_trainer.get_test_labels(),
                'test_proba': multi_model_trainer.get_test_proba(),
                'test_preds': multi_model_trainer.get_test_preds(),
                'test_metrics': multi_model_trainer.get_test_metrics(),
                'fold_thresholds': multi_model_trainer.get_fold_thresholds(),
                'shap_values_test': multi_model_trainer.get_shap_values_test(),
            }
            
            # Feature importance 저장 (모델별)
            feature_importances = multi_model_trainer.get_feature_importances()
            feature_importance_dfs = {}
            for model_name in feature_importances.keys():
                if len(feature_importances[model_name]) > 0:
                    # Fold별 평균 feature importance
                    avg_importance = np.mean(feature_importances[model_name], axis=0)
                    feature_importance_dfs[model_name] = pd.DataFrame({
                        'feature': X_train_raw.columns,
                        'importance': avg_importance
                    }).sort_values(by='importance', ascending=False)
            
            save_dict['feature_importances'] = feature_importance_dfs
            
            # 모델 비교 결과도 저장
            save_dict['comparison_results'] = {
                'validation': multi_model_trainer.weighted_avg_metrics,
                'test': multi_model_trainer.weighted_avg_test_metrics
            }
            
            # 샘플링 방법별로 파일 저장
            save_path = results_dir / f"models_comparison_{sampling_name}.pkl"
            with open(save_path, 'wb') as f:
                pickle.dump(save_dict, f)
            
            print(f"✓ 결과 저장 완료: {save_path}")
            print(f"  저장된 모델: {list(multi_model_trainer.models_to_train)}")
            print(f"  Train 크기: {len(X_train_raw)}, Test 크기: {len(X_test_raw)}")
            print(f"  Train 클래스 분포: {y_train_raw.value_counts().to_dict()}")
            print(f"  Test 클래스 분포: {y_test_raw.value_counts().to_dict()}")
            
        except Exception as save_error:
            print(f"⚠️  결과 저장 중 오류 발생: {save_error}")
            import traceback
            traceback.print_exc()
            print(f"  실험은 완료되었지만 저장에 실패했습니다.")
        
    except Exception as e:
        print(f"\n❌ 오류 발생 ({sampling_name}): {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n\n{'='*80}")
print("모든 실험 완료!")
print(f"{'='*80}")
print(f"총 {len(SAMPLING_METHODS)}가지 샘플링 방법에 대한 실험 결과가 저장되었습니다.")
print(f"결과 저장 위치: {results_dir}")
