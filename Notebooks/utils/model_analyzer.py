import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

# Optional: SHAP (설치되어 있으면 사용)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def get_original_feature_names(preprocessor, original_feature_names):
    """
    전처리된 feature 이름을 원본 feature 이름으로 매핑
    
    Parameters:
    -----------
    preprocessor : ColumnTransformer
        전처리 파이프라인
    original_feature_names : list
        원본 feature 이름 리스트
    
    Returns:
    --------
    processed_feature_names : list
        전처리된 feature 이름 리스트 (원본 이름 매핑됨)
    """
    if isinstance(preprocessor, ColumnTransformer):
        # sklearn >= 1.0: get_feature_names_out() 사용
        if hasattr(preprocessor, 'get_feature_names_out'):
            try:
                processed_feature_names = preprocessor.get_feature_names_out(original_feature_names)
                return list(processed_feature_names)
            except Exception as e:
                print(f"⚠️  get_feature_names_out() 실패, 수동 매핑 시도: {e}")
        
        # 수동 매핑 (fallback)
        processed_feature_names = []
        feature_idx = 0
        
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'remainder' and transformer == 'passthrough':
                # remainder는 원본 그대로
                for col in columns:
                    if col in original_feature_names:
                        processed_feature_names.append(col)
                        feature_idx += 1
            elif transformer == 'drop':
                continue
            else:
                # 연속형 변수 (StandardScaler)
                if name == 'num':
                    for col in columns:
                        processed_feature_names.append(col)
                        feature_idx += 1
                # 범주형 변수 (OneHotEncoder)
                elif name == 'cat':
                    if hasattr(transformer, 'get_feature_names_out'):
                        # OneHotEncoder의 feature 이름 추출
                        cat_feature_names = transformer.get_feature_names_out(columns)
                        processed_feature_names.extend(cat_feature_names)
                        feature_idx += len(cat_feature_names)
                    else:
                        # sklearn < 1.0: 수동으로 카테고리 이름 생성
                        for col in columns:
                            col_idx = columns.index(col) if isinstance(columns, list) else list(columns).index(col)
                            categories = transformer.categories_[col_idx]
                            # drop='first'이므로 첫 번째 카테고리는 제외
                            for cat in categories[1:]:
                                processed_feature_names.append(f"{col}_{cat}")
                                feature_idx += 1
    else:
        # StandardScaler만 사용된 경우
        processed_feature_names = original_feature_names
    
    return processed_feature_names


def map_processed_to_original_features(feature_names_processed, preprocessor, original_feature_names):
    """
    전처리된 feature 이름 리스트를 원본 feature 이름으로 변환
    
    Parameters:
    -----------
    feature_names_processed : list
        전처리된 feature 이름 (예: ['Feature_0', 'Feature_1', ...])
    preprocessor : ColumnTransformer
        전처리 파이프라인
    original_feature_names : list
        원본 feature 이름 리스트
    
    Returns:
    --------
    mapped_names : list
        원본 feature 이름으로 매핑된 리스트
    """
    processed_feature_names = get_original_feature_names(preprocessor, original_feature_names)
    
    # 인덱스 기반으로 매핑
    mapped_names = []
    for i in range(len(feature_names_processed)):
        if i < len(processed_feature_names):
            mapped_names.append(processed_feature_names[i])
        else:
            # 매핑이 없으면 원본 이름 유지
            mapped_names.append(feature_names_processed[i])
    
    return mapped_names


def plot_confusion_matrix(y_true, y_pred, model_name, sampler_method=None, ax=None):
    """
    Confusion Matrix 시각화
    
    Parameters:
    -----------
    y_true : array
        실제 타겟 값
    y_pred : array
        예측 값
    model_name : str
        모델 이름
    sampler_method : str, optional
        샘플링 방법
    ax : matplotlib axis, optional
        subplot axis
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['ODD_CP=0', 'ODD_CP=1'],
                yticklabels=['ODD_CP=0', 'ODD_CP=1'])
    
    title = f'Confusion Matrix - {model_name}'
    if sampler_method:
        title += f' ({sampler_method})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    # 정확도, 정밀도, 재현율 계산
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tn + tp) / (tn + fp + fn + tp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 텍스트 추가
    textstr = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}'
    ax.text(0.5, -0.15, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return ax


def plot_feature_importance(model, feature_names, model_name, sampler_method=None, top_n=20, ax=None, 
                           preprocessor=None, original_feature_names=None):
    """
    Feature Importance 시각화
    
    Parameters:
    -----------
    model : trained model
        학습된 모델
    feature_names : list
        feature 이름 리스트 (전처리된)
    model_name : str
        모델 이름
    sampler_method : str, optional
        샘플링 방법
    top_n : int
        상위 N개 feature만 표시
    ax : matplotlib axis, optional
        subplot axis
    preprocessor : ColumnTransformer, optional
        전처리 파이프라인 (원본 이름 매핑용)
    original_feature_names : list, optional
        원본 feature 이름 리스트
    """
    # 모델 타입에 따라 feature importance 추출
    importance = None
    
    if hasattr(model, 'feature_importances_'):
        # Tree-based models (RandomForest, GradientBoosting, XGBoost, LightGBM, CatBoost 등)
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models (LogisticRegression 등)
        importance = np.abs(model.coef_[0])
    elif model_name == 'SVC':
        # SVC는 feature importance가 없으므로 스킵
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'SVC does not provide feature importance', 
                ha='center', va='center', fontsize=14)
        ax.set_title(f'Feature Importance - {model_name}' + (f' ({sampler_method})' if sampler_method else ''))
        return ax
    else:
        # 다른 모델들 (KNN, NaiveBayes 등)
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'{model_name} does not provide feature importance', 
                ha='center', va='center', fontsize=14)
        ax.set_title(f'Feature Importance - {model_name}' + (f' ({sampler_method})' if sampler_method else ''))
        return ax
    
    if importance is None or len(importance) == 0:
        return ax
    
    # 원본 feature 이름으로 매핑 (가능한 경우)
    display_feature_names = feature_names[:len(importance)]
    if preprocessor is not None and original_feature_names is not None:
        try:
            display_feature_names = map_processed_to_original_features(
                feature_names[:len(importance)], 
                preprocessor, 
                original_feature_names
            )
        except Exception as e:
            print(f"⚠️  Feature 이름 매핑 중 오류 (원본 이름 사용): {e}")
            display_feature_names = feature_names[:len(importance)]
    
    # DataFrame 생성
    importance_df = pd.DataFrame({
        'feature': display_feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Bar plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
    bars = ax.barh(range(len(importance_df)), importance_df['importance'], color=colors)
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    
    title = f'Feature Importance (Top {top_n}) - {model_name}'
    if sampler_method:
        title += f' ({sampler_method})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 값 표시
    for i, (idx, row) in enumerate(importance_df.iterrows()):
        ax.text(row['importance'], i, f' {row["importance"]:.4f}', 
                va='center', fontsize=9)
    
    ax.invert_yaxis()
    plt.tight_layout()
    
    return ax, importance_df


def calculate_shap_values(model, X_train, X_test, model_name, sampler_method=None, max_samples=100):
    """
    SHAP values 계산 및 시각화
    
    Parameters:
    -----------
    model : trained model
        학습된 모델
    X_train : array or DataFrame
        학습 데이터
    X_test : array or DataFrame
        테스트 데이터
    model_name : str
        모델 이름
    sampler_method : str, optional
        샘플링 방법
    max_samples : int
        SHAP 계산에 사용할 최대 샘플 수
    
    Returns:
    --------
    shap_values : array
        SHAP values
    shap_explainer : explainer object
        SHAP explainer
    """
    if not SHAP_AVAILABLE:
        print("⚠️  SHAP이 설치되어 있지 않습니다. pip install shap 으로 설치해주세요.")
        return None, None
    
    # 샘플링 (너무 많으면 시간이 오래 걸림)
    if len(X_test) > max_samples:
        np.random.seed(42)
        sample_idx = np.random.choice(len(X_test), max_samples, replace=False)
        X_test_sample = X_test[sample_idx] if isinstance(X_test, np.ndarray) else X_test.iloc[sample_idx]
    else:
        X_test_sample = X_test
    
    try:
        # 모델 타입에 따라 적절한 explainer 선택
        if model_name in ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest', 
                          'GradientBoosting', 'ExtraTrees', 'DecisionTree', 'AdaBoost']:
            explainer = shap.TreeExplainer(model)
        elif model_name == 'LogisticRegression':
            explainer = shap.LinearExplainer(model, X_train[:100])  # 샘플링
        elif model_name == 'SVC':
            explainer = shap.KernelExplainer(model.predict_proba, X_train[:100])
        else:
            # 기본적으로 KernelExplainer 사용
            explainer = shap.KernelExplainer(model.predict_proba, X_train[:100])
        
        shap_values = explainer.shap_values(X_test_sample)
        
        # Binary classification인 경우 shap_values가 리스트일 수 있음
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # 클래스 1에 대한 SHAP values
        
        return shap_values, explainer, X_test_sample
        
    except Exception as e:
        print(f"⚠️  SHAP 계산 중 오류 발생: {e}")
        return None, None, None


def plot_shap_summary(shap_values, X_test_sample, feature_names, model_name, sampler_method=None, max_display=20):
    """
    SHAP Summary Plot
    
    Parameters:
    -----------
    shap_values : array
        SHAP values
    X_test_sample : array or DataFrame
        테스트 데이터 샘플
    feature_names : list
        feature 이름 리스트
    model_name : str
        모델 이름
    sampler_method : str, optional
        샘플링 방법
    max_display : int
        최대 표시 feature 수
    """
    if shap_values is None:
        return
    
    if isinstance(X_test_sample, pd.DataFrame):
        X_test_sample = X_test_sample.values
    
    # Feature names 설정
    if len(feature_names) != X_test_sample.shape[1]:
        feature_names = [f'Feature_{i}' for i in range(X_test_sample.shape[1])]
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_sample, 
                     feature_names=feature_names[:X_test_sample.shape[1]],
                     max_display=max_display, show=False)
    
    title = f'SHAP Summary Plot - {model_name}'
    if sampler_method:
        title += f' ({sampler_method})'
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


def plot_shap_bar(shap_values, feature_names, model_name, sampler_method=None, max_display=20):
    """
    SHAP Bar Plot (평균 절대값)
    
    Parameters:
    -----------
    shap_values : array
        SHAP values
    feature_names : list
        feature 이름 리스트
    model_name : str
        모델 이름
    sampler_method : str, optional
        샘플링 방법
    max_display : int
        최대 표시 feature 수
    """
    if shap_values is None:
        return
    
    # 평균 절대값 계산
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # DataFrame 생성
    shap_df = pd.DataFrame({
        'feature': feature_names[:len(mean_abs_shap)],
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False).head(max_display)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.plasma(np.linspace(0, 1, len(shap_df)))
    bars = ax.barh(range(len(shap_df)), shap_df['mean_abs_shap'], color=colors)
    ax.set_yticks(range(len(shap_df)))
    ax.set_yticklabels(shap_df['feature'])
    ax.set_xlabel('Mean |SHAP value|', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    
    title = f'SHAP Feature Importance (Top {max_display}) - {model_name}'
    if sampler_method:
        title += f' ({sampler_method})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 값 표시
    for i, (idx, row) in enumerate(shap_df.iterrows()):
        ax.text(row['mean_abs_shap'], i, f' {row["mean_abs_shap"]:.4f}', 
                va='center', fontsize=9)
    
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return shap_df


def analyze_best_model(sampler_result_dict, sampler_results, trainer_dict, 
                      X_test_orig, y_test_orig, feature_names, 
                      plot_confusion=True, plot_importance=True, plot_shap=True):
    """
    각 샘플링 방법별 최고 성능 모델 분석
    
    Parameters:
    -----------
    sampler_result_dict : dict
        샘플링 방법별 결과 딕셔너리
    sampler_results : dict
        샘플링 결과 딕셔너리
    trainer_dict : dict
        샘플링 방법별 trainer 딕셔너리
    X_test_orig : DataFrame
        원본 테스트 데이터
    y_test_orig : Series
        원본 테스트 타겟
    feature_names : list
        원본 feature 이름 리스트
    plot_confusion : bool
        Confusion Matrix 플롯 여부
    plot_importance : bool
        Feature Importance 플롯 여부
    plot_shap : bool
        SHAP 플롯 여부
    """
    results_summary = []
    
    for method in sampler_result_dict.keys():
        print(f"\n{'='*60}")
        print(f"📊 샘플링 방법: {method.upper()}")
        print(f"{'='*60}")
        
        # 최고 성능 모델 찾기 (F1 기준)
        results_df = sampler_result_dict[method]
        best_model_row = results_df.loc[results_df['F1'].idxmax()]
        best_model_name = best_model_row['Model']
        best_model = best_model_row['model']
        
        print(f"최고 성능 모델: {best_model_name}")
        print(f"  Accuracy: {best_model_row['Accuracy']:.4f}")
        print(f"  Precision: {best_model_row['Precision']:.4f}")
        print(f"  Recall: {best_model_row['Recall']:.4f}")
        print(f"  F1: {best_model_row['F1']:.4f}")
        
        # 예측값 가져오기
        y_pred = best_model_row['cls_df']['y_pred'].values
        
        # Trainer에서 전처리된 데이터 가져오기
        trainer = trainer_dict[method]
        X_test_processed = trainer.X_test_processed
        X_train_processed = trainer.X_train_processed
        
        # 1. Confusion Matrix
        if plot_confusion:
            print("\n📋 Confusion Matrix:")
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_confusion_matrix(y_test_orig, y_pred, best_model_name, method, ax)
            plt.show()
        
        # 2. Feature Importance
        if plot_importance:
            print("\n📊 Feature Importance:")
            try:
                # 전처리된 feature 이름 가져오기
                processed_feature_names = [f'Feature_{i}' for i in range(X_test_processed.shape[1])]
                fig, ax = plt.subplots(figsize=(10, 8))
                ax, importance_df = plot_feature_importance(
                    best_model, processed_feature_names, best_model_name, 
                    method, top_n=20, ax=ax,
                    preprocessor=trainer.preprocessor,  # 전처리 파이프라인 전달
                    original_feature_names=feature_names  # 원본 feature 이름 전달
                )
                plt.show()
                print(f"\n상위 10개 중요 변수:")
                print(importance_df.head(10)[['feature', 'importance']].to_string(index=False))
            except Exception as e:
                print(f"⚠️  Feature Importance 계산 중 오류: {e}")
                import traceback
                traceback.print_exc()
        
        # 3. SHAP Values
        if plot_shap:
            print("\n🔍 SHAP Values:")
            try:
                shap_values, explainer, X_test_sample = calculate_shap_values(
                    best_model, X_train_processed, X_test_processed, 
                    best_model_name, method, max_samples=100
                )
                
                if shap_values is not None:
                    processed_feature_names = [f'Feature_{i}' for i in range(X_test_processed.shape[1])]
                    
                    # 원본 feature 이름으로 매핑
                    try:
                        mapped_feature_names = map_processed_to_original_features(
                            processed_feature_names,
                            trainer.preprocessor,
                            feature_names
                        )
                    except Exception as e:
                        print(f"⚠️  SHAP Feature 이름 매핑 중 오류 (원본 이름 사용): {e}")
                        mapped_feature_names = processed_feature_names
                    
                    # SHAP Summary Plot
                    plot_shap_summary(shap_values, X_test_sample, 
                                    mapped_feature_names, best_model_name, method)
                    
                    # SHAP Bar Plot
                    shap_df = plot_shap_bar(shap_values, mapped_feature_names, 
                                           best_model_name, method)
                    print(f"\n상위 10개 SHAP 중요 변수:")
                    print(shap_df.head(10)[['feature', 'mean_abs_shap']].to_string(index=False))
            except Exception as e:
                print(f"⚠️  SHAP 계산 중 오류: {e}")
        
        results_summary.append({
            'sampler_method': method,
            'best_model': best_model_name,
            'accuracy': best_model_row['Accuracy'],
            'precision': best_model_row['Precision'],
            'recall': best_model_row['Recall'],
            'f1': best_model_row['F1']
        })
    
    # 전체 요약
    print(f"\n{'='*60}")
    print("📊 전체 요약")
    print(f"{'='*60}")
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))
    
    return summary_df

