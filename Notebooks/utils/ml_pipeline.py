from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 평가 함수 - best threshold에서 평가 결과 도출
import numpy as np

def evaluate_model(model, X_val, y_true):
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # best threshold 찾기 (기준: F1 score 최대)
    thresholds = np.arange(0.0, 1.01, 0.01)
    f1_scores = [f1_score(y_true, (y_pred_proba >= t).astype(int), zero_division=0) for t in thresholds]
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    # best threshold로 이진화
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'ROC AUC Score': roc_auc_score(y_true, y_pred_proba),
        'Best Threshold': best_threshold
    }


from scipy.special import softmax
def optimize_weights_and_temperature(metrics, test_proba, model_name='CatBoost', T_candidates = np.logspace(-3, 1, num=10), 
                                     weight_methods=['softmax', 'linear', 'uniform']):
    """
    T 값과 가중치 계산 방법을 최적화하여 최적의 예측 확률을 반환하는 함수
    :param metrics: Fold별 성능 메트릭 딕셔너리
    :param test_proba: Fold별 테스트 데이터 예측 확률
    :param model_name: 모델 이름 (기본값: 'CatBoost')
    :param T_candidates: 시도할 T 값 후보 리스트
    :param weight_methods: 시도할 가중치 계산 방법 리스트
    :return: 최적의 예측 확률과 선택된 파라미터
    """
    roc_auc_scores = np.array([fold_metrics['ROC AUC Score'] for fold_metrics in metrics[model_name]])
    test_proba_array = np.array(test_proba[model_name])
    best_pred_proba = None
    best_score = -np.inf
    best_params = None

    for T in T_candidates:
        for method in weight_methods:
            if method == 'softmax':
                # Softmax 기반 가중치 계산
                exp_scores = np.exp(roc_auc_scores / T)
                weights = exp_scores / np.sum(exp_scores)
            elif method == 'linear':
                # 선형 정규화 기반 가중치 계산
                weights = roc_auc_scores / np.sum(roc_auc_scores)
            elif method == 'uniform':
                # 균일 가중치
                weights = np.ones(len(roc_auc_scores)) / len(roc_auc_scores)

            # 가중 평균 계산
            pred_proba = np.average(test_proba_array, axis=0, weights=weights)
            # Validation ROC AUC 기준으로 최적화 (임시로 Fold 평균 ROC AUC 사용)
            weighted_avg_roc_auc = np.average(roc_auc_scores, weights=weights)

            if weighted_avg_roc_auc > best_score:
                best_score = weighted_avg_roc_auc
                best_pred_proba = pred_proba
                best_params = {'T': T, 'method': method, 'weights': weights}

    print(f"최적 파라미터: T={best_params['T']}, Method={best_params['method']}")
    print(f"최적 가중치: {best_params['weights'].tolist()}")
    print(f"최적화된 평균 ROC AUC: {best_score:.6f}")
    return best_pred_proba, best_params
# def train_model():
    