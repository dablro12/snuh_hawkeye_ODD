import pandas as pd 
import warnings
import numpy as np
import pickle
from pathlib import Path
import os 
import json 

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

variable_dict = json.load(open("/workspace/data/variable_category.json", "r"))

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

# ============================================================================
# Gender별 데이터 분리 (전처리 전에 수행)
# ============================================================================
print(f"\n{'='*80}")
print("Gender별 데이터 분리")
print(f"{'='*80}")

# Gender별로 데이터 분리
gender_train_data = {}
gender_test_data = {}

# 전체 (A)
gender_train_data['A'] = train_df.copy()
gender_test_data['A'] = test_df.copy()

# 남성 (M) - Sex == 1
gender_train_data['M'] = train_df[train_df['Sex'] == 1].copy()
gender_test_data['M'] = test_df[test_df['Sex'] == 1].copy()

# 여성 (FM) - Sex == 2 (원본 데이터에서 여성은 2로 인코딩됨)
gender_train_data['FM'] = train_df[train_df['Sex'] == 2].copy()
gender_test_data['FM'] = test_df[test_df['Sex'] == 2].copy()

for gender_key in ['A', 'M', 'FM']:
    print(f"\nGender: {gender_key}")
    print(f"  Train: {len(gender_train_data[gender_key])}개 샘플")
    print(f"  Test: {len(gender_test_data[gender_key])}개 샘플")
    if len(gender_train_data[gender_key]) > 0:
        print(f"  Train 클래스 분포: {gender_train_data[gender_key]['ODD_CP'].value_counts().to_dict()}")
    if len(gender_test_data[gender_key]) > 0:
        print(f"  Test 클래스 분포: {gender_test_data[gender_key]['ODD_CP'].value_counts().to_dict()}")

# ============================================================================
# 2. 실험 설정 (9가지 조합: Feature Set 3가지 × Gender 3가지)
# ============================================================================
from Notebooks.utils.data_preprocessor import check_preprocessing_needs, preprocess_dataframe
from Notebooks.utils.data_preprocessor import data_preprocess_pipeline
from Notebooks.utils.data_preprocessor import extract_features_with_onehot
from Notebooks.utils.ml_model import MultiModelFoldTrainer

# 샘플링 방법 정의 (Downsample + SMOTEEN만 사용)
sampling_name = 'downsample_SMOTEEN'
sampling_config = {
    'type': 'oversample',
    'params': {'train_size_per_class': 240, 'method': 'SMOTEENN'}
}

# 실험 설정 테이블
EXPERIMENTS = [
    # Feature Set, Gender, Feature File, Exp Name
    ('A', 'A', '/workspace/data/results/variable_total.csv', '4_EXP_Methods_A_A'),      # 부모+청소년 / 전체
    ('A', 'M', '/workspace/data/results/variable_total.csv', '4_EXP_Methods_A_M'),      # 부모+청소년 / 남
    ('A', 'FM', '/workspace/data/results/variable_total.csv', '4_EXP_Methods_A_FM'),  # 부모+청소년 / 여
    ('P', 'A', '/workspace/data/results/variable_parent.csv', '4_EXP_Methods_P_A'),     # 부모 / 전체
    ('P', 'M', '/workspace/data/results/variable_parent.csv', '4_EXP_Methods_P_M'),     # 부모 / 남
    ('P', 'FM', '/workspace/data/results/variable_parent.csv', '4_EXP_Methods_P_FM'),  # 부모 / 여
    ('Ad', 'A', '/workspace/data/results/variable_adolescent.csv', '4_EXP_Methods_Ad_A'),   # 청소년 / 전체
    ('Ad', 'M', '/workspace/data/results/variable_adolescent.csv', '4_EXP_Methods_Ad_M'),   # 청소년 / 남
    ('Ad', 'FM', '/workspace/data/results/variable_adolescent.csv', '4_EXP_Methods_Ad_FM'), # 청소년 / 여
]

# 결과 저장 디렉토리 생성
results_dir = Path("/workspace/data/results/variable_gender_comp")
results_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 3. 9가지 실험 실행
# ============================================================================
print(f"\n\n{'='*80}")
print(f"총 {len(EXPERIMENTS)}개 실험 시작")
print(f"{'='*80}\n")

# Gender별 전처리된 데이터 저장 (캐싱용)
gender_X_train = {}
gender_y_train = {}
gender_X_test = {}
gender_y_test = {}

for exp_idx, (feature_set, gender, _, exp_name) in enumerate(EXPERIMENTS, 1):
    print(f"\n\n{'#'*80}")
    print(f"# 실험 {exp_idx}/{len(EXPERIMENTS)}: {exp_name}")
    print(f"# Feature Set: {feature_set}, Gender: {gender}")
    print(f"{'#'*80}\n")
    
    try:
        # 1. variable features 추출 (전처리 전에 수행)
        # variable_category.json 파일 참고하여 name 값만 추출
        if feature_set == 'A':
            # 부모+청소년: 둘 다 포함
            variable_features = [v['name'] for v in variable_dict['adolescent_variables']] + \
                                [v['name'] for v in variable_dict['parent_variables']]
        elif feature_set == 'P':
            # 부모만
            variable_features = [v['name'] for v in variable_dict['parent_variables']]
        elif feature_set == 'Ad':
            # 청소년만
            variable_features = [v['name'] for v in variable_dict['adolescent_variables']]
        
        print(f"\n{'='*80}")
        print(f"Variable Features (원본)")
        print(f"{'='*80}")
        print(variable_features)
        
        # 2. Gender별 데이터 가져오기 및 Feature 선택 (전처리 전)
        train_df_gender = gender_train_data[gender]
        test_df_gender = gender_test_data[gender]
        
        if len(train_df_gender) == 0 or len(test_df_gender) == 0:
            print(f"  경고: Gender {gender} 데이터가 비어있습니다. 실험을 건너뜁니다.")
            continue
        
        # 전처리 전에 feature 선택 (원본 컬럼명 사용)
        selected_features_raw = extract_features_with_onehot(variable_features, train_df_gender.columns)
        
        print(f"\n{'='*80}")
        print(f"선택된 Features (원핫인코딩 포함, 총 {len(selected_features_raw)}개)")
        print(f"{'='*80}")
        print(selected_features_raw)
        
        # 3. Gender별 데이터 전처리 (선택된 feature만 사용, 캐싱되어 있지 않은 경우에만 수행)
        cache_key = f"{gender}_{feature_set}"
        if cache_key not in gender_X_train:
            print(f"\n{'='*80}")
            print(f"Gender {gender} (Feature Set: {feature_set}) 데이터 전처리 시작")
            print(f"{'='*80}\n")
            
            # 선택된 feature만 포함하여 데이터 준비
            train_df_gender_selected = train_df_gender[selected_features_raw + ['ODD_CP']].copy()
            test_df_gender_selected = test_df_gender[selected_features_raw + ['ODD_CP']].copy()
            
            # Train 데이터 전처리
            X_train_gender = train_df_gender_selected.drop(columns=['ODD_CP'])
            y_train_gender = train_df_gender_selected['ODD_CP']
            
            X_train_gender = preprocess_dataframe(
                X_train_gender, 
                target_col='ODD_CP',
                drop_weight=True,
                convert_categorical=['Answ', 'IGD_P', 'FEdu', 'MEdu', 'FJob', 'MJob', 'Age_Grp', 'P_Marr'],
                convert_ordinal=['ST1', 'ST2', 'ST3', 'ST4', 'PAF', 'MAlc', 'FAlc', "MTob", "FTob", "MAlc", "FAlc", "GAlc", "MTob", "FTob", "GTob"], 
                convert_binary=['SRD_CP', 'IGD_P', 'Sex', 'PSleep', 'SBV', 'SBP', 'CBV', 'CBP', 'GDec', 'BF', 'RFG', 'MentD', 'AdolSlp', 'MoodD', 'AnxD'],
                drop_low_variance=False,
                drop_leakage=True
            )
            X_train_gender = data_preprocess_pipeline(X_train_gender)
            
            # Test 데이터 전처리
            X_test_gender = test_df_gender_selected.drop(columns=['ODD_CP'])
            y_test_gender = test_df_gender_selected['ODD_CP']
            
            X_test_gender = preprocess_dataframe(
                X_test_gender, 
                target_col='ODD_CP',
                drop_weight=True,
                convert_categorical=['Answ', 'IGD_P', 'FEdu', 'MEdu', 'FJob', 'MJob', 'Age_Grp', 'P_Marr'],
                convert_ordinal=['ST1', 'ST2', 'ST3', 'ST4', 'PAF', 'MAlc', 'FAlc', "MTob", "FTob", "MAlc", "FAlc", "GAlc", "MTob", "FTob", "GTob"], 
                convert_binary=['SRD_CP', 'IGD_P', 'Sex', 'PSleep', 'SBV', 'SBP', 'CBV', 'CBP', 'GDec', 'BF', 'RFG', 'MentD', 'AdolSlp', 'MoodD', 'AnxD'],
                drop_low_variance=False,
                drop_leakage=True
            )
            X_test_gender = data_preprocess_pipeline(X_test_gender)
            
            # 컬럼 일치 확인 및 정렬
            common_cols = list(set(X_train_gender.columns) & set(X_test_gender.columns))
            X_train_gender = X_train_gender[common_cols]
            X_test_gender = X_test_gender[common_cols]
            
            # Gender별로 분리했으므로 Sex 컬럼 제거 (상수이거나 불필요)
            if 'Sex' in X_train_gender.columns:
                X_train_gender = X_train_gender.drop(columns=['Sex'])
                print(f"  ✅ Sex 컬럼 제거 (Gender별 분리 후 불필요)")
            if 'Sex' in X_test_gender.columns:
                X_test_gender = X_test_gender.drop(columns=['Sex'])
            
            # 컬럼 다시 일치 확인 (Sex 제거 후)
            common_cols = list(set(X_train_gender.columns) & set(X_test_gender.columns))
            X_train_gender = X_train_gender[common_cols]
            X_test_gender = X_test_gender[common_cols]
            
            # 저장 (캐싱)
            gender_X_train[cache_key] = X_train_gender
            gender_y_train[cache_key] = y_train_gender
            gender_X_test[cache_key] = X_test_gender
            gender_y_test[cache_key] = y_test_gender
            
            print(f"  ✅ Gender {gender} (Feature Set: {feature_set}) 전처리 완료")
            print(f"    Train: {len(X_train_gender)}개 샘플, {len(X_train_gender.columns)}개 변수")
            print(f"    Test: {len(X_test_gender)}개 샘플, {len(X_test_gender.columns)}개 변수")
            print(f"    Train 클래스 분포: {y_train_gender.value_counts().to_dict()}")
            print(f"    Test 클래스 분포: {y_test_gender.value_counts().to_dict()}")
        else:
            print(f"  ✅ Gender {gender} (Feature Set: {feature_set}) 전처리된 데이터 사용 (캐싱됨)")
        
        # 4. 전처리된 데이터 가져오기
        X_train_filtered = gender_X_train[cache_key]
        y_train_filtered = gender_y_train[cache_key]
        X_test_filtered = gender_X_test[cache_key]
        y_test_filtered = gender_y_test[cache_key]
        
        print(f"\nGender 필터링 완료 ({gender}):")
        print(f"  Train: {len(X_train_filtered)}개 샘플")
        print(f"  Test: {len(X_test_filtered)}개 샘플")
        print(f"  Train 클래스 분포: {y_train_filtered.value_counts().to_dict()}")
        print(f"  Test 클래스 분포: {y_test_filtered.value_counts().to_dict()}")
        
        # 선택된 features만 사용 (전처리 후 컬럼명이 변경되었을 수 있으므로 모든 컬럼 사용)
        X_train_variable = X_train_filtered
        X_test_variable = X_test_filtered
        
        # 전처리 후 최종 선택된 features (컬럼명이 변경되었을 수 있음)
        selected_features = list(X_train_variable.columns)
        
        print(f"\n✅ X_train_variable shape: {X_train_variable.shape}")
        print(f"✅ X_test_variable shape: {X_test_variable.shape}")
        print(f"✅ y_train shape: {y_train_filtered.shape}")
        print(f"✅ y_test shape: {y_test_filtered.shape}")

        
        # 5. 모델 학습 (LightGBM만 사용)
        # MultiModelFoldTrainer 내부에서 K-Fold 분할 후 각 fold의 train 부분에만 샘플링 적용
        print(f"\n{'='*80}")
        print(f"모델 학습 시작: LightGBM")
        print(f"샘플링 방법: {sampling_name}")
        print(f"{'='*80}\n")
        
        multi_model_trainer = MultiModelFoldTrainer(
            models_to_train=['LightGBM'],  # LightGBM만 사용
            n_splits=5, 
            random_state=42, 
            T=0.01,
            sampling_config=sampling_config  # 샘플링 설정 전달
        )
        
        # 학습 및 평가 (원본 데이터 전달, MultiModelFoldTrainer 내부에서 샘플링 처리)
        multi_model_trainer.fit(
            X = X_train_variable, y = y_train_filtered, 
            X_test = X_test_variable, y_test = y_test_filtered
        )
        
        # 5. 결과 저장
        print(f"\n{'='*80}")
        print(f"결과 저장 중: {exp_name}")
        print(f"{'='*80}\n")
        
        save_dict = {
            'exp_name': exp_name,
            'feature_set': feature_set,
            'gender': gender,
            'sampling_method': sampling_name,
            'sampling_config': sampling_config,
            'selected_features': selected_features,
            'variable_features': variable_features,
            'test_inputs': X_test_variable,
            'test_labels': multi_model_trainer.get_test_labels(),
            'test_proba': multi_model_trainer.get_test_proba(),
            'test_preds': multi_model_trainer.get_test_preds(),
            'test_metrics': multi_model_trainer.get_test_metrics(),
            'fold_thresholds': multi_model_trainer.get_fold_thresholds(),
            'shap_values_test': multi_model_trainer.get_shap_values_test(),
        }
        
        # Feature importance 저장
        feature_importances = multi_model_trainer.get_feature_importances()
        feature_importance_dfs = {}
        for model_name in feature_importances.keys():
            if len(feature_importances[model_name]) > 0:
                # Fold별 평균 feature importance
                avg_importance = np.mean(feature_importances[model_name], axis=0)
                feature_importance_dfs[model_name] = pd.DataFrame({
                    'feature': X_train_variable.columns,
                    'importance': avg_importance
                }).sort_values(by='importance', ascending=False)
        
        save_dict['feature_importances'] = feature_importance_dfs
        
        # 모델 비교 결과도 저장
        save_dict['comparison_results'] = {
            'validation': multi_model_trainer.weighted_avg_metrics,
            'test': multi_model_trainer.weighted_avg_test_metrics
        }
        
        # 결과 저장
        save_path = results_dir / f"{exp_name}.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"✓ 결과 저장 완료: {save_path}")
        print(f"  저장된 모델: {list(multi_model_trainer.models_to_train)}")
        print(f"  Feature Set: {feature_set}, Gender: {gender}")
        print(f"  Train 크기: {len(X_train_variable)}, Test 크기: {len(X_test_variable)}")
        print(f"  Train 클래스 분포: {y_train_filtered.value_counts().to_dict()}")
        print(f"  Test 클래스 분포: {y_test_filtered.value_counts().to_dict()}")
        
    except Exception as e:
        print(f"\n❌ 오류 발생 ({exp_name}): {e}")
        import traceback
        traceback.print_exc()
        print(f"  실험을 건너뛰고 다음 실험을 진행합니다.\n")
        continue

print(f"\n\n{'='*80}")
print("모든 실험 완료!")
print(f"{'='*80}")
print(f"총 {len(EXPERIMENTS)}개 실험 중 완료된 실험 결과가 저장되었습니다.")
print(f"모델: LightGBM")
print(f"샘플링 방법: {sampling_name}")
print(f"결과 저장 위치: {results_dir}")
