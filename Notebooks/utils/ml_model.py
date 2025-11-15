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
from IPython import display
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



from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import mode
import numpy as np
import shap
import pandas as pd

# 모델 import
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available")

class MultiModelFoldTrainer:
    """여러 모델을 비교하는 K-Fold Cross Validation Trainer"""
    
    def __init__(self, models_to_train=None, n_splits=5, random_state=123, T=0.01, sampling_config=None):
        """
        Parameters:
        -----------
        models_to_train : list, optional
            학습할 모델 이름 리스트. None이면 모든 가능한 모델 사용
            가능한 모델: 'CatBoost', 'XGBoost', 'LightGBM', 'RandomForest', 
                        'GradientBoosting', 'LogisticRegression', 'SVM', 'MLP'
        n_splits : int
            K-Fold 분할 수
        random_state : int
            랜덤 시드
        T : float
            Softmax temperature parameter
        sampling_config : dict, optional
            샘플링 설정. None이면 샘플링 적용 안 함
            예: {'type': 'oversample', 'params': {'train_size_per_class': 240, 'method': 'SMOTE'}}
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.T = T
        self.sampling_config = sampling_config
        
        # 기본 모델 설정
        self.model_configs = self._get_model_configs()
        
        # 학습할 모델 선택
        if models_to_train is None:
            self.models_to_train = list(self.model_configs.keys())
        else:
            self.models_to_train = [m for m in models_to_train if m in self.model_configs]
            if not self.models_to_train:
                raise ValueError(f"사용 가능한 모델이 없습니다. 가능한 모델: {list(self.model_configs.keys())}")
        
        # 결과 저장용 딕셔너리 초기화
        self.metrics = {model: [] for model in self.models_to_train}
        self.test_metrics = {model: [] for model in self.models_to_train}
        self.feature_importances = {model: [] for model in self.models_to_train}
        self.test_proba = {model: [] for model in self.models_to_train}
        self.test_preds = {model: [] for model in self.models_to_train}
        self.fold_thresholds = {model: [] for model in self.models_to_train}
        self.shap_values_train = {model: [] for model in self.models_to_train}
        self.shap_values_test = {model: [] for model in self.models_to_train}
        
        self.fold_weights = {}
        self.weighted_avg_metrics = {}
        self.weighted_avg_test_metrics = {}
        self.y_test = None
    
    def _get_model_configs(self):
        """모델 설정 반환"""
        configs = {}
        
        # CatBoost 설정
        configs['CatBoost'] = {
            'class': CatBoostClassifier,
            'params': dict(
                iterations=1000, learning_rate=0.38577, depth=8, 
                l2_leaf_reg=9.587765, subsample=0.748324, random_strength=0.0, 
                class_weights=[1, 10], min_data_in_leaf=59, 
                leaf_estimation_iterations=1, loss_function='Logloss', 
                eval_metric='AUC', verbose=False, random_seed=self.random_state
            ),
            'has_shap': True
        }
        
        # XGBoost 설정
        if XGBOOST_AVAILABLE:
            configs['XGBoost'] = {
                'class': xgb.XGBClassifier,
                'params': dict(
                    n_estimators=1000, learning_rate=0.1, max_depth=8,
                    subsample=0.8, colsample_bytree=0.8, 
                    scale_pos_weight=10, random_state=self.random_state,
                    eval_metric='auc', use_label_encoder=False
                ),
                'has_shap': True
            }
        
        # LightGBM 설정
        if LIGHTGBM_AVAILABLE:
            configs['LightGBM'] = {
                'class': lgb.LGBMClassifier,
                'params': dict(
                    n_estimators=1000, learning_rate=0.1, max_depth=8,
                    subsample=0.8, colsample_bytree=0.8,
                    class_weight={0: 1, 1: 10}, random_state=self.random_state,
                    verbose=-1
                ),
                'has_shap': True
            }
        
        # Random Forest 설정
        configs['RandomForest'] = {
            'class': RandomForestClassifier,
            'params': dict(
                n_estimators=200, max_depth=15, min_samples_split=5,
                min_samples_leaf=2, class_weight={0: 1, 1: 10},
                random_state=self.random_state, n_jobs=-1
            ),
            'has_shap': True
        }
        
        # Gradient Boosting 설정
        configs['GradientBoosting'] = {
            'class': GradientBoostingClassifier,
            'params': dict(
                n_estimators=200, learning_rate=0.1, max_depth=8,
                min_samples_split=5, min_samples_leaf=2,
                random_state=self.random_state
            ),
            'has_shap': True
        }
        
        # Logistic Regression 설정
        configs['LogisticRegression'] = {
            'class': LogisticRegression,
            'params': dict(
                C=1.0, class_weight={0: 1, 1: 10},
                random_state=self.random_state, max_iter=1000,
                solver='lbfgs', n_jobs=-1
            ),
            'has_shap': True
        }
        
        # SVM 설정
        configs['SVM'] = {
            'class': SVC,
            'params': dict(
                C=1.0, kernel='rbf', probability=True,
                class_weight={0: 1, 1: 10}, random_state=self.random_state
            ),
            'has_shap': False  # SVM은 SHAP 계산이 느림
        }
        
        # MLP (Neural Network) 설정
        configs['MLP'] = {
            'class': MLPClassifier,
            'params': dict(
                hidden_layer_sizes=(100, 50), activation='relu',
                solver='adam', alpha=0.0001, learning_rate='adaptive',
                max_iter=500, random_state=self.random_state, early_stopping=True
            ),
            'has_shap': False  # MLP는 SHAP 계산이 복잡함
        }
        
        return configs
    
    def _create_model(self, model_name):
        """모델 인스턴스 생성"""
        config = self.model_configs[model_name]
        return config['class'](**config['params'])
    
    def _get_feature_importance(self, model, model_name):
        """모델별 feature importance 추출"""
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'get_feature_importance'):
            return model.get_feature_importance()
        else:
            return None
    
    def _get_shap_values(self, model, X, model_name):
        """SHAP values 계산"""
        try:
            config = self.model_configs.get(model_name, {})
            if not config.get('has_shap', False):
                return None
            
            # Tree-based models
            if model_name in ['CatBoost', 'XGBoost', 'LightGBM', 'RandomForest', 'GradientBoosting']:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                if isinstance(shap_values, list):
                    return shap_values[1]  # binary classification: class 1
                return shap_values
            
            # Linear models (Logistic Regression)
            elif model_name == 'LogisticRegression':
                explainer = shap.LinearExplainer(model, X)
                shap_values = explainer.shap_values(X)
                if isinstance(shap_values, list):
                    return shap_values[1]  # binary classification: class 1
                return shap_values
            
        except Exception as e:
            print(f"Warning: SHAP values 계산 실패 ({model_name}): {e}")
            return None
        return None
    
    def _apply_sampling_to_fold_train(self, X_train_fold, y_train_fold):
        """
        각 fold의 train 부분에만 샘플링 적용
        Validation set은 원본 그대로 사용하므로 여기서는 train 부분만 처리
        """
        if self.sampling_config is None:
            return X_train_fold, y_train_fold
        
        sampling_type = self.sampling_config['type']
        params = self.sampling_config['params'].copy()
        params['random_state'] = self.random_state
        
        if sampling_type == 'downsample':
            # Downsampling: 클래스 0을 n_train_class0개로 제한
            n_train_class0 = params.get('n_train_class0')
            if n_train_class0 is None:
                # None이면 그대로 사용
                return X_train_fold, y_train_fold
            
            # 클래스별 분리
            df_train = pd.concat([X_train_fold, y_train_fold], axis=1)
            df_0 = df_train[df_train.iloc[:, -1] == 0]
            df_1 = df_train[df_train.iloc[:, -1] == 1]
            
            # 클래스 0 다운샘플링
            if len(df_0) > n_train_class0:
                df_0 = df_0.sample(n=n_train_class0, random_state=self.random_state)
            
            # 클래스 1은 그대로 사용
            df_train_final = pd.concat([df_0, df_1], axis=0).sample(frac=1, random_state=self.random_state).reset_index(drop=True)
            X_train_resampled = df_train_final.iloc[:, :-1]
            y_train_resampled = df_train_final.iloc[:, -1]
            
            return X_train_resampled, y_train_resampled
        
        elif sampling_type == 'oversample':
            # Oversampling: SMOTE 등을 사용하여 클래스 1을 오버샘플링
            train_size_per_class = params.get('train_size_per_class', 240)
            method = params.get('method', 'SMOTE')
            
            from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
            from imblearn.combine import SMOTETomek, SMOTEENN
            
            # 클래스별 분리
            df_train = pd.concat([X_train_fold, y_train_fold], axis=1)
            target_col = df_train.columns[-1]
            df_0 = df_train[df_train[target_col] == 0]
            df_1 = df_train[df_train[target_col] == 1]
            
            # 클래스 0 처리: train_size_per_class개로 맞춤 (단, 실제 데이터 크기를 초과하지 않도록)
            # 각 fold의 train 부분은 원본의 일부이므로, train_size_per_class가 실제 크기보다 클 수 있음
            target_size_0 = min(train_size_per_class, len(df_0))
            
            if len(df_0) < target_size_0:
                # 실제로는 이 경우는 발생하지 않지만 안전을 위해
                df_0_final = df_0.copy()
            elif len(df_0) > target_size_0:
                df_0_final = df_0.sample(n=target_size_0, random_state=self.random_state)
            else:
                df_0_final = df_0.copy()
            
            # 클래스 1 처리: SMOTE 등을 사용하여 train_size_per_class개로 오버샘플링
            # 클래스 1의 경우, 클래스 0과 비율을 맞추기 위해 오버샘플링
            target_size_1 = train_size_per_class
            
            if len(df_1) < target_size_1:
                # 클래스 0과 1을 합쳐서 SMOTE 적용
                df_temp = pd.concat([df_0_final, df_1], axis=0)
                X_temp = df_temp.drop(columns=[target_col])
                y_temp = df_temp[target_col]
                
                # k_neighbors 체크: 클래스 1이 너무 적으면 SMOTE가 실패할 수 있음
                k_neighbors = min(5, max(1, len(df_1) - 1))
                n_neighbors = min(5, max(1, len(df_1) - 1))
                
                method_upper = method.upper()
                try:
                    if 'SMOTEEN' in method_upper:
                        sampler = SMOTEENN(
                            sampling_strategy={0: len(df_0_final), 1: target_size_1},
                            random_state=self.random_state
                        )
                    elif 'SMOTETOMEK' in method_upper:
                        sampler = SMOTETomek(
                            sampling_strategy={0: len(df_0_final), 1: target_size_1},
                            random_state=self.random_state
                        )
                    elif 'ADASYN' in method_upper:
                        sampler = ADASYN(
                            sampling_strategy={0: len(df_0_final), 1: target_size_1},
                            random_state=self.random_state,
                            n_neighbors=n_neighbors
                        )
                    else:  # SMOTE
                        sampler = SMOTE(
                            sampling_strategy={0: len(df_0_final), 1: target_size_1},
                            random_state=self.random_state,
                            k_neighbors=k_neighbors
                        )
                    
                    X_resampled, y_resampled = sampler.fit_resample(X_temp, y_temp)
                    df_resampled = pd.concat([
                        pd.DataFrame(X_resampled, columns=X_train_fold.columns),
                        pd.Series(y_resampled, name=target_col)
                    ], axis=1)
                    
                    # 클래스별 추출
                    df_0_final = df_resampled[df_resampled[target_col] == 0]
                    df_1_final = df_resampled[df_resampled[target_col] == 1]
                    
                except Exception as e:
                    # SMOTE 실패 시 RandomOverSampler로 대체
                    print(f"    ⚠️  {method} 샘플링 실패 (클래스 1: {len(df_1)}개), RandomOverSampler로 대체: {e}")
                    ros = RandomOverSampler(
                        sampling_strategy={0: len(df_0_final), 1: target_size_1},
                        random_state=self.random_state
                    )
                    X_resampled, y_resampled = ros.fit_resample(X_temp, y_temp)
                    df_resampled = pd.concat([
                        pd.DataFrame(X_resampled, columns=X_train_fold.columns),
                        pd.Series(y_resampled, name=target_col)
                    ], axis=1)
                    df_0_final = df_resampled[df_resampled[target_col] == 0]
                    df_1_final = df_resampled[df_resampled[target_col] == 1]
                    
            elif len(df_1) > target_size_1:
                df_1_final = df_1.sample(n=target_size_1, random_state=self.random_state)
            else:
                df_1_final = df_1.copy()
            
            # 최종 train set 구성
            df_train_final = pd.concat([df_0_final, df_1_final], axis=0).sample(frac=1, random_state=self.random_state).reset_index(drop=True)
            X_train_resampled = df_train_final.drop(columns=[target_col])
            y_train_resampled = df_train_final[target_col]
            
            return X_train_resampled, y_train_resampled
        
        # 샘플링이 없거나 알 수 없는 타입인 경우 원본 반환
        return X_train_fold, y_train_fold
    
    def fit(self, X, y, X_test, y_test=None):
        """모든 모델 학습 및 평가"""
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        self.y_test = y_test if y_test is not None else None
        
        # 모든 결과 초기화
        for model_name in self.models_to_train:
            self.metrics[model_name].clear()
            self.test_metrics[model_name].clear()
            self.feature_importances[model_name].clear()
            self.test_proba[model_name].clear()
            self.test_preds[model_name].clear()
            self.fold_thresholds[model_name].clear()
            self.shap_values_train[model_name].clear()
            self.shap_values_test[model_name].clear()
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"\n{'='*60}")
            print(f"Fold {fold}/{self.n_splits}")
            print(f"{'='*60}")
            
            # 원본 train/val 분할
            X_train_orig, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train_orig, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 각 fold의 train 부분에만 샘플링 적용 (validation은 원본 그대로)
            X_train, y_train = self._apply_sampling_to_fold_train(X_train_orig, y_train_orig)
            
            print(f"  Train (원본): {len(X_train_orig)}개 → Train (샘플링 후): {len(X_train)}개")
            print(f"  Validation (원본 그대로): {len(X_val)}개")
            print(f"  Train 클래스 분포: {pd.Series(y_train).value_counts().to_dict()}")
            print(f"  Val 클래스 분포: {pd.Series(y_val).value_counts().to_dict()}")
            
            for model_name in self.models_to_train:
                print(f"\n--- {model_name} ---")
                
                # 모델 생성 및 학습
                model = self._create_model(model_name)
                model.fit(X_train, y_train)
                
                # Validation 예측
                val_proba = model.predict_proba(X_val)[:, 1]
                
                # Best threshold 찾기 (F1 기준)
                thresholds = np.linspace(0, 1, 200)
                f1s = [f1_score(y_val, (val_proba >= t).astype(int), zero_division=0) for t in thresholds]
                best_idx = np.argmax(f1s)
                best_threshold = thresholds[best_idx]
                val_pred_best = (val_proba >= best_threshold).astype(int)
                
                # Validation 메트릭
                val_metrics = {
                    'Accuracy': accuracy_score(y_val, val_pred_best),
                    'Precision': precision_score(y_val, val_pred_best, zero_division=0),
                    'Recall': recall_score(y_val, val_pred_best, zero_division=0),
                    'F1 Score': f1_score(y_val, val_pred_best, zero_division=0),
                    'ROC AUC Score': roc_auc_score(y_val, val_proba),
                    'Best_Threshold': best_threshold
                }
                self.metrics[model_name].append(val_metrics)
                
                # Feature importance
                feature_imp = self._get_feature_importance(model, model_name)
                if feature_imp is not None:
                    self.feature_importances[model_name].append(feature_imp)
                
                # SHAP values
                shap_train = self._get_shap_values(model, X_train, model_name)
                shap_test = self._get_shap_values(model, X_test, model_name)
                if shap_train is not None:
                    self.shap_values_train[model_name].append(shap_train)
                if shap_test is not None:
                    self.shap_values_test[model_name].append(shap_test)
                
                # Test 예측
                test_proba = model.predict_proba(X_test)[:, 1]
                test_pred = (test_proba >= best_threshold).astype(int)
                self.test_proba[model_name].append(test_proba)
                self.test_preds[model_name].append(test_pred)
                self.fold_thresholds[model_name].append(best_threshold)
                
                # Test 메트릭
                if y_test is not None:
                    try:
                        test_metrics_fold = {
                            'Accuracy': accuracy_score(y_test, test_pred),
                            'Precision': precision_score(y_test, test_pred, zero_division=0),
                            'Recall': recall_score(y_test, test_pred, zero_division=0),
                            'F1 Score': f1_score(y_test, test_pred, zero_division=0),
                            'ROC AUC Score': roc_auc_score(y_test, test_proba),
                            'Best_Threshold': best_threshold
                        }
                    except Exception as e:
                        test_metrics_fold = {
                            'Accuracy': np.nan, 'Precision': np.nan, 'Recall': np.nan,
                            'F1 Score': np.nan, 'ROC AUC Score': np.nan,
                            'Best_Threshold': best_threshold
                        }
                    self.test_metrics[model_name].append(test_metrics_fold)
                
                print(f"  Val F1: {val_metrics['F1 Score']:.4f}, Val AUC: {val_metrics['ROC AUC Score']:.4f}")
                if y_test is not None:
                    print(f"  Test F1: {test_metrics_fold['F1 Score']:.4f}, Test AUC: {test_metrics_fold['ROC AUC Score']:.4f}")
        
        # 결과 출력 (에러가 발생해도 저장은 진행되도록 try-except 처리)
        try:
            self.print_comparison_results()
        except Exception as e:
            print(f"\n⚠️  결과 출력 중 오류 발생: {e}")
            print("  (학습은 완료되었으며 결과는 저장됩니다)")
        
        return self
    
    def calc_softmax_weights(self, model_name):
        """모델별 fold weights 계산"""
        f1_scores = np.array([m['F1 Score'] for m in self.metrics[model_name]])
        exp_scores = np.exp(f1_scores / self.T)
        weights = exp_scores / np.sum(exp_scores)
        self.fold_weights[model_name] = weights
        return weights
    
    def calculate_weighted_metrics(self, model_name):
        """모델별 weighted 평균 메트릭"""
        weights = self.calc_softmax_weights(model_name)
        metric_keys = [k for k in self.metrics[model_name][0] if k != 'Best_Threshold']
        weighted_metrics = {
            metric: sum(w * m[metric] for w, m in zip(weights, self.metrics[model_name]))
            for metric in metric_keys
        }
        self.weighted_avg_metrics[model_name] = weighted_metrics
        return weighted_metrics
    
    def calculate_weighted_test_metrics(self, model_name):
        """모델별 weighted test 메트릭"""
        if len(self.test_metrics[model_name]) == 0:
            return None
        weights = self.calc_softmax_weights(model_name)
        metric_keys = [k for k in self.test_metrics[model_name][0] if k != 'Best_Threshold']
        weighted_metrics = {
            metric: sum(w * m[metric] for w, m in zip(weights, self.test_metrics[model_name]))
            for metric in metric_keys
        }
        self.weighted_avg_test_metrics[model_name] = weighted_metrics
        return weighted_metrics
    
    def print_comparison_results(self):
        """모든 모델 비교 결과 출력"""
        print("\n" + "="*80)
        print("모델 비교 결과 (Weighted Average)")
        print("="*80)
        
        # Validation 결과 비교
        print("\n[Validation Set]")
        print("-"*80)
        comparison_data = []
        for model_name in self.models_to_train:
            weighted_metrics = self.calculate_weighted_metrics(model_name)
            comparison_data.append({
                'Model': model_name,
                **weighted_metrics
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1 Score', ascending=False)
        print(comparison_df.to_string(index=False))
        
        # Test 결과 비교
        if self.y_test is not None:
            print("\n[Test Set]")
            print("-"*80)
            test_comparison_data = []
            for model_name in self.models_to_train:
                weighted_test_metrics = self.calculate_weighted_test_metrics(model_name)
                if weighted_test_metrics:
                    test_comparison_data.append({
                        'Model': model_name,
                        **weighted_test_metrics
                    })
            
            if test_comparison_data:
                test_comparison_df = pd.DataFrame(test_comparison_data)
                test_comparison_df = test_comparison_df.sort_values('F1 Score', ascending=False)
                print(test_comparison_df.to_string(index=False))
        
        # Best 모델 출력
        best_model = comparison_df.iloc[0]['Model']
        print(f"\n🏆 Best Model (Validation F1): {best_model}")
        print(f"   F1 Score: {comparison_df.iloc[0]['F1 Score']:.4f}")
        print(f"   ROC AUC: {comparison_df.iloc[0]['ROC AUC Score']:.4f}")
    
    # Getter 메서드들
    def get_val_metrics(self):
        return self.metrics
    
    def get_test_metrics(self):
        return self.test_metrics
    
    def get_feature_importances(self):
        return self.feature_importances
    
    def get_test_labels(self):
        return self.y_test
    
    def get_test_proba(self):
        return self.test_proba
    
    def get_test_preds(self):
        return self.test_preds
    
    def get_fold_thresholds(self):
        return self.fold_thresholds
    
    def get_shap_values_train(self):
        return self.shap_values_train
    
    def get_shap_values_test(self):
        return self.shap_values_test

# 사용 예시: 모든 모델 비교
# models_to_train=None이면 모든 가능한 모델 사용
# 또는 특정 모델만 선택: ['CatBoost', 'RandomForest', 'LogisticRegression'] 등
# 
# 주의: 아래 코드는 모듈 import 시 실행되지 않도록 주석 처리됨
# 노트북이나 스크립트에서 직접 사용할 때만 주석을 해제하여 사용하세요
#
# multi_model_trainer = MultiModelFoldTrainer(
#     models_to_train=None,  # None이면 모든 가능한 모델 사용
#     # models_to_train=['CatBoost', 'XGBoost', 'LightGBM', 'RandomForest', 'GradientBoosting', 'LogisticRegression'],  # 특정 모델만 선택
#     n_splits=5, 
#     random_state=42, 
#     T=0.01
# )
# multi_model_trainer.fit(X_train, y_train, X_test, y_test=y_test)