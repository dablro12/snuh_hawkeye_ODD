from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from tqdm import tqdm

# Optional: XGBoost, LightGBM, CatBoost (설치되어 있으면 사용)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

class ClassifierTrainer:
    """
    Binary Classification Trainer for ODD_CP prediction
    X: 입력 feature (DataFrame 또는 array) - 연속형과 범주형 변수 모두 포함 가능
    y: Binary target (0 또는 1)
    """
    def __init__(self, X, y, X_test=None, y_test=None, random_state=42, n_estimators=500, epoch=100, lr=0.05, n_jobs=-1):
        """
        Parameters:
        -----------
        X : DataFrame or array
            학습용 feature (또는 전체 데이터)
        y : Series or array
            학습용 타겟 변수 (또는 전체 데이터)
        X_test : DataFrame or array, optional
            테스트용 feature (제공되면 X는 train set으로 간주)
        y_test : Series or array, optional
            테스트용 타겟 변수 (제공되면 y는 train set으로 간주)
        """
        self.n_estimators = n_estimators
        self.epoch = epoch
        self.lr = lr
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.device = "cpu"
        
        # X가 DataFrame인지 확인
        if isinstance(X, pd.DataFrame):
            self.X_df = X.copy()
            self.feature_names = X.columns.tolist()
        else:
            # numpy array인 경우 DataFrame으로 변환
            self.X_df = pd.DataFrame(X)
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            self.X_df.columns = self.feature_names
        
        # y를 numpy array로 변환 (binary: 0 또는 1)
        self.y = np.asarray(y).astype(int)
        
        # Binary classification 확인
        unique_classes = np.unique(self.y)
        if len(unique_classes) != 2 or not all(c in [0, 1] for c in unique_classes):
            raise ValueError(f"y는 binary classification이어야 합니다 (0 또는 1). 현재 클래스: {unique_classes}")
        
        # Test set이 제공된 경우 (이미 분리된 경우)
        if X_test is not None and y_test is not None:
            if isinstance(X_test, pd.DataFrame):
                self.X_test = X_test.copy()
            else:
                self.X_test = pd.DataFrame(X_test, columns=self.feature_names)
            self.y_test = np.asarray(y_test).astype(int)
            self.X_train = self.X_df
            self.y_train = self.y
            
            print(f"Train 타겟 변수 분포: {np.bincount(self.y_train)} (클래스 0: {np.sum(self.y_train==0)}, 클래스 1: {np.sum(self.y_train==1)})")
            print(f"Test 타겟 변수 분포: {np.bincount(self.y_test)} (클래스 0: {np.sum(self.y_test==0)}, 클래스 1: {np.sum(self.y_test==1)})")
        else:
            # Test set이 제공되지 않은 경우: 자동으로 분리
            print(f"타겟 변수 분포: {np.bincount(self.y)} (클래스 0: {np.sum(self.y==0)}, 클래스 1: {np.sum(self.y==1)})")
            
            # 데이터 분리
            (
                self.X_train, self.X_test,
                self.y_train, self.y_test
            ) = train_test_split(
                self.X_df, self.y, test_size=0.2,
                random_state=self.random_state,
                stratify=self.y
            )
            print(f"데이터 자동 분리 완료 (Train/Test: {len(self.X_train)} / {len(self.X_test)})")
        
        # 데이터 누수(Data Leakage) 감지 (train set에 대해서만)
        self._detect_data_leakage()
        
        # 범주형/연속형 변수 구분
        self._identify_column_types()
        
        # 전처리 파이프라인 구성
        self._build_preprocessor()
        
        # 전처리 적용 (train으로 fit, test로 transform)
        self.X_train_processed = self.preprocessor.fit_transform(self.X_train)
        self.X_test_processed = self.preprocessor.transform(self.X_test)
        
        print(f"전처리 완료 ✅")
        print(f"  - Train/Test: {len(self.X_train)} / {len(self.X_test)}")
        print(f"  - 원본 feature 수: {len(self.feature_names)}")
        print(f"  - 전처리 후 feature 수: {self.X_train_processed.shape[1]}")
        print(f"  - 범주형 변수: {len(self.categorical_cols)}개")
        print(f"  - 연속형 변수: {len(self.numeric_cols)}개")
        print("-" * 60)
    
    def _detect_data_leakage(self):
        """데이터 누수(Data Leakage) 변수 감지 (train set에 대해서만)"""
        leakage_vars = []
        numeric_cols = self.X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            # 타겟 변수와의 일치율 계산 (train set 기준)
            match_rate = (self.X_train[col] == self.y_train).mean()
            # 상관관계 계산
            corr = abs(self.X_train[col].corr(pd.Series(self.y_train)))
            
            # 일치율이 95% 이상이거나 상관관계가 0.9 이상이면 경고
            if match_rate >= 0.95 or corr >= 0.9:
                leakage_vars.append({
                    'variable': col,
                    'match_rate': match_rate,
                    'correlation': corr
                })
        
        if leakage_vars:
            print("\n⚠️  데이터 누수(Data Leakage) 경고:")
            print("다음 변수들이 타겟 변수와 매우 높은 상관관계를 가지고 있습니다:")
            for var in leakage_vars:
                print(f"  - {var['variable']}: 일치율 {var['match_rate']:.2%}, 상관관계 {var['correlation']:.4f}")
            print("이 변수들을 제거하는 것을 권장합니다.\n")
    
    def _identify_column_types(self):
        """범주형과 연속형 변수 자동 구분 (train set 기준)"""
        self.categorical_cols = self.X_train.select_dtypes(include=['object']).columns.tolist()
        self.numeric_cols = self.X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        # 범주형 변수가 없으면 빈 리스트로 설정
        if not self.categorical_cols:
            self.categorical_cols = []
    
    def _build_preprocessor(self):
        """전처리 파이프라인 구성"""
        transformers = []
        
        # 연속형 변수: StandardScaler
        if self.numeric_cols:
            transformers.append(('num', StandardScaler(), self.numeric_cols))
        
        # 범주형 변수: OneHotEncoder
        if self.categorical_cols:
            transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
                                self.categorical_cols))
        
        if transformers:
            self.preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough'
            )
        else:
            # 변수가 없는 경우 (이상한 경우)
            self.preprocessor = StandardScaler()
        
    def print_cls_results(self, y_test, y_pred, model_name):
        """Binary classification 결과 출력"""
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        acc = accuracy_score(y_test, y_pred)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        df = pd.DataFrame({
            "Y_True": y_test,
            "Y_Pred": y_pred
        })
        
        print(f"\n🧩 Binary Classification ({model_name})")
        print(df.head(10).to_string(index=False))
        print(f"\n📊 Metrics:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"\n📋 Confusion Matrix:")
        print(f"              Predicted")
        print(f"              0     1")
        print(f"  Actual 0  {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"         1  {cm[1,0]:4d}  {cm[1,1]:4d}")
        print("=" * 60)

    def get_models(self):
        """Binary classification 모델들 반환"""
        models = {
            # 기본 모델들
            "LogisticRegression": LogisticRegression(
                max_iter=self.epoch, 
                random_state=self.random_state,
                class_weight='balanced'
            ),
            "SVC": SVC(
                kernel="rbf", 
                C=1.0, 
                probability=True, 
                random_state=self.random_state,
                class_weight='balanced'
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=self.n_estimators, 
                random_state=self.random_state, 
                n_jobs=self.n_jobs,
                class_weight='balanced'
            ),
            "GradientBoosting": GradientBoostingClassifier(
                random_state=self.random_state,
                n_estimators=self.n_estimators,
                learning_rate=self.lr
            ),
            "MLP": MLPClassifier(
                hidden_layer_sizes=(256, 128), 
                max_iter=self.epoch, 
                random_state=self.random_state
            ),
            
            # 추가 모델들
            "AdaBoost": AdaBoostClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.lr,
                random_state=self.random_state
            ),
            "ExtraTrees": ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                class_weight='balanced'
            ),
            "DecisionTree": DecisionTreeClassifier(
                random_state=self.random_state,
                class_weight='balanced'
            ),
            "KNN": KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            ),
            "NaiveBayes": GaussianNB()
        }
        
        # XGBoost (설치되어 있으면 추가)
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.lr,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                eval_metric='logloss',
                use_label_encoder=False
            )
        
        # LightGBM (설치되어 있으면 추가)
        if LIGHTGBM_AVAILABLE:
            models["LightGBM"] = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.lr,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                class_weight='balanced',
                verbose=-1
            )
        
        # CatBoost (설치되어 있으면 추가)
        if CATBOOST_AVAILABLE:
            models["CatBoost"] = CatBoostClassifier(
                iterations=self.n_estimators,
                learning_rate=self.lr,
                random_state=self.random_state,
                verbose=False,
                thread_count=self.n_jobs
            )
        
        return models
    
    def list_available_models(self):
        """사용 가능한 모델 목록 출력"""
        models = self.get_models()
        print("=" * 60)
        print("📋 사용 가능한 모델 목록")
        print("=" * 60)
        print(f"총 {len(models)}개 모델:")
        for i, name in enumerate(models.keys(), 1):
            print(f"  {i:2d}. {name}")
        
        # Optional 모델 상태
        print("\n📦 Optional 모델 상태:")
        print(f"  - XGBoost: {'✅ 사용 가능' if XGBOOST_AVAILABLE else '❌ 설치 필요 (pip install xgboost)'}")
        print(f"  - LightGBM: {'✅ 사용 가능' if LIGHTGBM_AVAILABLE else '❌ 설치 필요 (pip install lightgbm)'}")
        print(f"  - CatBoost: {'✅ 사용 가능' if CATBOOST_AVAILABLE else '❌ 설치 필요 (pip install catboost)'}")
        print("=" * 60)
        return list(models.keys())

    def run_all(self, printf=True):
        """모든 모델 학습 및 평가 - x를 넣으면 training 후 결과 반환"""
        models = self.get_models()
        results = []
        self.trained_models = {}  # 학습된 모델 저장

        for name, clf in tqdm(models.items(), desc="Classifier별 진행", ncols=80,
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
            if printf:
                print(f"\n🚀 {name} 모델 학습 시작...")
                print("-" * 60)

            # 전처리된 데이터로 학습
            clf.fit(self.X_train_processed, self.y_train)
            y_pred = np.array(clf.predict(self.X_test_processed)).ravel()

            # 학습된 모델 저장
            self.trained_models[name] = clf

            # Binary classification metrics
            acc = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='binary', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='binary', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='binary', zero_division=0)

            cls_df = pd.DataFrame({
                "y_true": self.y_test,
                "y_pred": y_pred
            })

            if printf:
                self.print_cls_results(self.y_test, y_pred, name)

            results.append({
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1": round(f1, 4),
                "cls_df": cls_df,
                "model": clf  # 모델 객체도 저장
            })

        df_result = pd.DataFrame(results).sort_values("F1", ascending=False)
        if printf:
            print("\n📊 전체 분류 모델 비교 결과 요약")
            print(df_result[["Model", "Accuracy", "Precision", "Recall", "F1"]].to_string(index=False))
        return df_result
    
    def get_model(self, model_name):
        """학습된 모델 가져오기 (run_all() 실행 후 사용 가능)"""
        if not hasattr(self, 'trained_models') or not self.trained_models:
            raise ValueError("먼저 run_all()을 실행하여 모델을 학습시켜주세요.")
        
        if model_name not in self.trained_models:
            raise ValueError(f"모델 '{model_name}'을 찾을 수 없습니다. 사용 가능한 모델: {list(self.trained_models.keys())}")
        
        return self.trained_models[model_name]
    
    def predict_proba(self, X_new, model_name=None):
        """새로운 데이터에 대한 예측 확률 (run_all() 실행 후 사용 가능)"""
        if not hasattr(self, 'trained_models') or not self.trained_models:
            raise ValueError("먼저 run_all()을 실행하여 모델을 학습시켜주세요.")
        
        if model_name is None:
            raise ValueError("model_name을 지정해주세요. 사용 가능한 모델: " + ", ".join(self.trained_models.keys()))
        
        if model_name not in self.trained_models:
            raise ValueError(f"모델 '{model_name}'을 찾을 수 없습니다. 사용 가능한 모델: {list(self.trained_models.keys())}")
        
        # X_new가 DataFrame이 아니면 변환
        if not isinstance(X_new, pd.DataFrame):
            X_new = pd.DataFrame(X_new, columns=self.feature_names)
        
        # 전처리 적용
        X_new_processed = self.preprocessor.transform(X_new)
        
        # 학습된 모델로 예측 확률 계산
        model = self.trained_models[model_name]
        y_pred_proba = model.predict_proba(X_new_processed)[:, 1]  # 클래스 1의 확률
        
        return y_pred_proba
    
    def predict(self, X_new, model_name=None):
        """새로운 데이터에 대한 예측 (run_all() 실행 후 사용 가능)"""
        if not hasattr(self, 'trained_models') or not self.trained_models:
            raise ValueError("먼저 run_all()을 실행하여 모델을 학습시켜주세요.")
        
        if model_name is None:
            raise ValueError("model_name을 지정해주세요. 사용 가능한 모델: " + ", ".join(self.trained_models.keys()))
        
        if model_name not in self.trained_models:
            raise ValueError(f"모델 '{model_name}'을 찾을 수 없습니다. 사용 가능한 모델: {list(self.trained_models.keys())}")
        
        # X_new가 DataFrame이 아니면 변환
        if not isinstance(X_new, pd.DataFrame):
            X_new = pd.DataFrame(X_new, columns=self.feature_names)
        
        # 전처리 적용
        X_new_processed = self.preprocessor.transform(X_new)
        
        # 학습된 모델로 예측
        model = self.trained_models[model_name]
        y_pred = model.predict(X_new_processed)
        y_pred_proba = model.predict_proba(X_new_processed)[:, 1]  # 클래스 1의 확률
        
        return y_pred, y_pred_proba


class SoftVotingClassifierTrainer(ClassifierTrainer):
    """
    Soft Voting Classifier Trainer
    여러 모델의 예측 확률을 평균내어 최종 예측을 수행
    """
    def __init__(self, X, y, X_test=None, y_test=None, random_state=42, n_estimators=500, epoch=100, lr=0.05, n_jobs=-1):
        """ClassifierTrainer와 동일한 초기화"""
        super().__init__(X, y, X_test, y_test, random_state, n_estimators, epoch, lr, n_jobs)
        self.voting_models = {}  # Voting에 사용할 모델들 저장
    
    def run_all(self, printf=True):
        """모든 모델 학습 후 Soft Voting으로 최종 예측"""
        models = self.get_models()
        self.trained_models = {}  # 학습된 모델 저장
        self.voting_models = {}  # Voting에 사용할 모델들 (predict_proba 지원하는 모델만)
        
        # 1단계: 모든 모델 학습
        if printf:
            print("=" * 60)
            print("1단계: 개별 모델 학습")
            print("=" * 60)
        
        for name, clf in tqdm(models.items(), desc="개별 모델 학습", ncols=80,
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
            try:
                # 전처리된 데이터로 학습
                clf.fit(self.X_train_processed, self.y_train)
                
                # 학습된 모델 저장
                self.trained_models[name] = clf
                
                # predict_proba를 지원하는 모델만 voting에 포함
                if hasattr(clf, 'predict_proba'):
                    self.voting_models[name] = clf
                    
            except Exception as e:
                if printf:
                    print(f"⚠️  {name} 모델 학습 실패: {e}")
                continue
        
        if printf:
            print(f"\n✅ 총 {len(self.trained_models)}개 모델 학습 완료")
            print(f"✅ Soft Voting에 {len(self.voting_models)}개 모델 사용")
            print("=" * 60)
        
        # 2단계: Soft Voting으로 예측
        if printf:
            print("\n2단계: Soft Voting 예측")
            print("=" * 60)
        
        # 모든 모델의 예측 확률 평균 계산
        y_proba_sum = np.zeros(len(self.y_test))
        
        for name, clf in self.voting_models.items():
            try:
                y_proba = clf.predict_proba(self.X_test_processed)[:, 1]  # 클래스 1의 확률
                y_proba_sum += y_proba
            except Exception as e:
                if printf:
                    print(f"⚠️  {name} 예측 확률 계산 실패: {e}")
                continue
        
        # 평균 확률 계산
        y_proba_avg = y_proba_sum / len(self.voting_models)
        
        # 임계값 0.5로 최종 예측
        y_pred = (y_proba_avg >= 0.5).astype(int)
        
        # 3단계: 평가
        acc = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(self.y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average='binary', zero_division=0)
        
        cls_df = pd.DataFrame({
            "y_true": self.y_test,
            "y_pred": y_pred
        })
        
        if printf:
            print(f"\n📊 Soft Voting 결과:")
            print(f"  Accuracy:  {acc:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"\n📋 Confusion Matrix:")
            cm = confusion_matrix(self.y_test, y_pred)
            print(f"              Predicted")
            print(f"              0     1")
            print(f"  Actual 0  {cm[0,0]:4d}  {cm[0,1]:4d}")
            print(f"         1  {cm[1,0]:4d}  {cm[1,1]:4d}")
            print("=" * 60)
        
        # 결과를 DataFrame 형태로 반환 (기존 형식과 호환)
        results = [{
            "Model": "SoftVoting",
            "Accuracy": round(acc, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            "cls_df": cls_df,
            "model": None  # Voting은 단일 모델 객체가 아님
        }]
        
        # 개별 모델 결과도 포함 (선택사항)
        if printf:
            print("\n📊 개별 모델 성능 비교:")
            individual_results = []
            for name, clf in self.trained_models.items():
                try:
                    y_pred_ind = clf.predict(self.X_test_processed)
                    acc_ind = accuracy_score(self.y_test, y_pred_ind)
                    precision_ind = precision_score(self.y_test, y_pred_ind, average='binary', zero_division=0)
                    recall_ind = recall_score(self.y_test, y_pred_ind, average='binary', zero_division=0)
                    f1_ind = f1_score(self.y_test, y_pred_ind, average='binary', zero_division=0)
                    
                    individual_results.append({
                        "Model": name,
                        "Accuracy": round(acc_ind, 4),
                        "Precision": round(precision_ind, 4),
                        "Recall": round(recall_ind, 4),
                        "F1": round(f1_ind, 4)
                    })
                except:
                    continue
            
            if individual_results:
                df_ind = pd.DataFrame(individual_results).sort_values("F1", ascending=False)
                print(df_ind[["Model", "Accuracy", "Precision", "Recall", "F1"]].to_string(index=False))
        
        df_result = pd.DataFrame(results)
        return df_result
    
    def predict_proba(self, X_new, model_name=None):
        """Soft Voting으로 예측 확률 계산"""
        if not hasattr(self, 'voting_models') or not self.voting_models:
            raise ValueError("먼저 run_all()을 실행하여 모델을 학습시켜주세요.")
        
        # model_name 파라미터는 무시 (Soft Voting은 모든 모델 사용)
        if not isinstance(X_new, pd.DataFrame):
            X_new = pd.DataFrame(X_new, columns=self.feature_names)
        
        # 전처리 적용
        X_new_processed = self.preprocessor.transform(X_new)
        
        # 모든 모델의 예측 확률 평균 계산
        y_proba_sum = np.zeros(len(X_new))
        
        for name, clf in self.voting_models.items():
            try:
                y_proba = clf.predict_proba(X_new_processed)[:, 1]
                y_proba_sum += y_proba
            except Exception as e:
                continue
        
        # 평균 확률 반환
        y_proba_avg = y_proba_sum / len(self.voting_models)
        return y_proba_avg
    
    def predict(self, X_new, model_name=None, threshold=0.5):
        """Soft Voting으로 예측"""
        y_proba = self.predict_proba(X_new, model_name)
        y_pred = (y_proba >= threshold).astype(int)
        return y_pred, y_proba

