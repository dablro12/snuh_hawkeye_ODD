import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def check_missing_col(dataframe):
    """
    Usage : missing_col = check_missing_col(train)
    """ 
    missing_col = []
    counted_missing_col = 0
    for i,col in enumerate(dataframe.columns):
        missing_values = sum(dataframe[col].isna())
        is_missing = True if missing_values >= 1 else False
        if is_missing:
            counted_missing_col +=1
            print('결측치가 있는 칼럼은: %s입니다.'%col)
            print('해당 칼럼에 총 %s개의 결측치가 존재합니다. '%missing_values)
            missing_col.append([col, dataframe[col].dtype])
    if counted_missing_col == 0:
        print('결측치가 존재하지 않습니다.')
    return missing_col

def ODD_CP_Sex_Analyzer(df):
    # Set style similar to Nature format (seaborn 'whitegrid' style with 'paper' context)
    sns.set(style="whitegrid", context="paper", font_scale=1.3)
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['legend.frameon'] = False

    # 1. Calculate group-wise, overall, and within-group ratios for ODD_CP x Sex
    summary_df = (
        df.groupby(['ODD_CP', 'Sex'])
        .size()
        .reset_index(name='count')
    )
    total = len(df)
    summary_df['percent_total'] = summary_df['count'] / total * 100

    # Count per ODD_CP group
    count_ODD = df['ODD_CP'].value_counts().sort_index()
    percent_ODD = count_ODD / total * 100

    # Sex ratio within each ODD_CP group
    odd1 = df[df['ODD_CP'] == 1]
    odd0 = df[df['ODD_CP'] == 0]
    n_odd1 = len(odd1)
    n_odd0 = len(odd0)

    cp1_sex = odd1['Sex'].value_counts().sort_index()
    cp0_sex = odd0['Sex'].value_counts().sort_index()
    cp1_sex_pct = cp1_sex / n_odd1 * 100
    cp0_sex_pct = cp0_sex / n_odd0 * 100

    # 2. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={'width_ratios': [1.7, 1, 1]})

    # (1) Grouped barplot for overall distribution of ODD_CP x Sex
    group_names = {0: 'ODD_CP=0', 1: 'ODD_CP=1'}
    sex_names = {1: 'Male', 2: 'Female'}
    plot_data = summary_df.copy()
    plot_data['ODD_CP'] = plot_data['ODD_CP'].map(group_names)
    plot_data['Sex'] = plot_data['Sex'].map(sex_names)

    bar1 = sns.barplot(
        data=plot_data,
        x='ODD_CP', y='count', hue='Sex',
        palette=['royalblue', '#e74c3c'],
        edgecolor='black', ax=axes[0]
    )
    axes[0].set_title('Sample count by ODD_CP and Sex', pad=14)
    axes[0].set_xlabel('ODD_CP Group')
    axes[0].set_ylabel('Sample Count')
    # Annotate N and % on top of each bar
    for idx, row in plot_data.iterrows():
        bar = bar1.patches[idx]
        axes[0].annotate(
            f"{int(row['count'])}\n({row['percent_total']:.2f}%)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            xytext=(0, 4), textcoords='offset points'
        )
    axes[0].legend(title="Sex", loc="upper right", frameon=False)

    # (2) Pie chart for overall ODD_CP group ratios
    pie_colors = ['#3498db', '#e74c3c']
    axes[1].pie(count_ODD, labels=[group_names[k] for k in count_ODD.index],
                autopct=lambda pct: f"{int(pct * total / 100)}\n({pct:.2f}%)",
                startangle=90, counterclock=False, colors=pie_colors, pctdistance=0.8,
                wedgeprops=dict(edgecolor='white', linewidth=2))
    axes[1].set_title("ODD_CP Distribution among All Participants", pad=12)

    # (3) Stacked bar chart for sex ratio within ODD_CP=1/0 groups
    width = 0.4
    axes[2].bar(['ODD_CP=1'], [cp1_sex_pct.get(1, 0)], width, label='Male', color='royalblue', edgecolor='black')
    axes[2].bar(['ODD_CP=1'], [cp1_sex_pct.get(2, 0)], width,
                bottom=[cp1_sex_pct.get(1, 0)], color='#e74c3c', edgecolor='black', label='Female')
    axes[2].bar(['ODD_CP=0'], [cp0_sex_pct.get(1, 0)], width, color='royalblue', edgecolor='black')
    axes[2].bar(['ODD_CP=0'], [cp0_sex_pct.get(2, 0)], width,
                bottom=[cp0_sex_pct.get(1, 0)], color='#e74c3c', edgecolor='black')
    # Text annotations
    axes[2].text(0, cp1_sex_pct.get(1, 0) / 2, f"{cp1_sex.get(1, 0)}\n({cp1_sex_pct.get(1, 0):.2f}%)", ha='center', va='center', color='white', fontsize=12, fontweight='bold')
    axes[2].text(0, cp1_sex_pct.get(1, 0) + cp1_sex_pct.get(2, 0) / 2, f"{cp1_sex.get(2, 0)}\n({cp1_sex_pct.get(2, 0):.2f}%)", ha='center', va='center', color='white', fontsize=12, fontweight='bold')
    axes[2].text(1, cp0_sex_pct.get(1, 0) / 2, f"{cp0_sex.get(1, 0)}\n({cp0_sex_pct.get(1, 0):.2f}%)", ha='center', va='center', color='white', fontsize=12, fontweight='bold')
    axes[2].text(1, cp0_sex_pct.get(1, 0) + cp0_sex_pct.get(2, 0) / 2, f"{cp0_sex.get(2, 0)}\n({cp0_sex_pct.get(2, 0):.2f}%)", ha='center', va='center', color='white', fontsize=12, fontweight='bold')
    axes[2].set_ylim(0, 100)
    axes[2].set_ylabel('Within Group Sex Ratio (%)')
    axes[2].set_title('Sex Ratio within ODD_CP Groups', pad=14)
    axes[2].set_xticks([0, 1])
    axes[2].set_xticklabels(['ODD_CP=1', 'ODD_CP=0'])
    axes[2].legend(['Male', 'Female'], loc='upper right')

    plt.tight_layout(w_pad=3)
    plt.show()


# df의 결측치 분석 결과를 print해주는 함수 정의
def print_missing_summary(df):
    missing_counts = df.isnull().sum()
    missing_rate = df.isnull().mean() * 100

    missing_summary = pd.DataFrame({
        'Missing Count': missing_counts,
        'Missing Rate (%)': missing_rate
    })

    print("결측치 컬럼별 개수 및 비율:")
    display(missing_summary[missing_summary['Missing Count'] > 0].sort_values('Missing Count', ascending=False))
    print("\n결측치가 있는 행 개수:", df.isnull().any(axis=1).sum())


def print_basic_statistics(df):
    """기본 통계량 요약"""
    print("=" * 60)
    print("데이터셋 기본 정보")
    print("=" * 60)
    print(f"총 샘플 수: {len(df):,}")
    print(f"총 변수 수: {len(df.columns)}")
    print(f"\n연속형 변수: {len(df.select_dtypes(include=[np.number]).columns)}개")
    print(f"범주형 변수: {len(df.select_dtypes(include=['object']).columns)}개")
    
    print("\n" + "=" * 60)
    print("타겟 변수(ODD_CP) 분포")
    print("=" * 60)
    if 'ODD_CP' in df.columns:
        target_dist = df['ODD_CP'].value_counts().sort_index()
        print(target_dist)
        print(f"\n비율:")
        print((target_dist / len(df) * 100).round(2))
    
    print("\n" + "=" * 60)
    print("주요 연속형 변수 통계량")
    print("=" * 60)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'ODD_CP' in numeric_cols:
        numeric_cols.remove('ODD_CP')
    display(df[numeric_cols].describe().T)


def analyze_age_group_distribution(df):
    """연령 그룹별 ODD_CP 분포 분석"""
    if 'Age_Grp' not in df.columns:
        print("Age_Grp 컬럼이 없습니다.")
        return
    
    sns.set(style="whitegrid", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 연령 그룹별 전체 분포
    age_dist = df['Age_Grp'].value_counts().sort_index()
    axes[0].bar(age_dist.index, age_dist.values, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Age Group')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Age Group Distribution')
    for i, v in enumerate(age_dist.values):
        axes[0].text(age_dist.index[i], v, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 연령 그룹별 ODD_CP 비율
    age_odd = pd.crosstab(df['Age_Grp'], df['ODD_CP'], normalize='index') * 100
    age_odd.plot(kind='bar', stacked=True, ax=axes[1], 
                 color=['#3498db', '#e74c3c'], edgecolor='black')
    axes[1].set_xlabel('Age Group')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].set_title('ODD_CP Distribution by Age Group')
    axes[1].legend(['ODD_CP=0', 'ODD_CP=1'], title='ODD_CP')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.show()
    
    print("\n연령 그룹별 ODD_CP 분포:")
    display(pd.crosstab(df['Age_Grp'], df['ODD_CP'], margins=True))


from IPython.display import display
def analyze_mental_health_variables(df):
    """정신건강 관련 변수와 ODD_CP의 관계 분석"""
    mental_vars = ['GAD', 'PHQ', 'SAS', 'Z_GAD', 'Z_PHQ', 'Z_SAS']
    available_vars = [v for v in mental_vars if v in df.columns]
    
    if not available_vars:
        print("정신건강 관련 변수를 찾을 수 없습니다.")
        return
    
    sns.set(style="whitegrid", context="paper", font_scale=1.2)
    n_vars = len(available_vars)
    cols = 3
    rows = (n_vars + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if n_vars > 1 else [axes]
    
    for idx, var in enumerate(available_vars):
        if df[var].dtype in ['int64', 'float64']:
            # 박스플롯
            df.boxplot(column=var, by='ODD_CP', ax=axes[idx], grid=False)
            axes[idx].set_title(f'{var} by ODD_CP')
            axes[idx].set_xlabel('ODD_CP')
            axes[idx].set_ylabel(var)
            axes[idx].get_figure().suptitle('')
    
    # 사용하지 않는 subplot 제거
    for idx in range(len(available_vars), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()
    
    # 통계 요약
    print("\n정신건강 변수별 ODD_CP 그룹 통계:")
    for var in available_vars:
        if df[var].dtype in ['int64', 'float64']:
            summary = df.groupby('ODD_CP')[var].describe()
            print(f"\n{var}:")
            display(summary)


def analyze_correlation_matrix(X, y):
    """타겟 변수와의 상관관계 행렬 시각화 (X, y 입력)"""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # X 입력이 DataFrame이 아니면 변환
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    # y를 Series로 변환
    y_array = y if isinstance(y, pd.Series) else pd.Series(y)
    # y 이름 지정
    if hasattr(y, "name") and y.name is not None:
        target_name = y.name
    else:
        target_name = "Target"

    # 수치형 컬럼만 추출
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("수치형 특성이 없습니다.")
        return

    # Corr matrix 생성: X의 수치형 컬럼 + y
    df_corr = X[numeric_cols].copy()
    df_corr[target_name] = y_array.values

    corr_matrix = df_corr.corr()

    # 타겟과의 상관관계만 볼 수도 있지만, 전체 행렬로 표시
    plt.figure(figsize=(len(corr_matrix)*0.7+3, len(corr_matrix)*0.7+3))
    sns.set(style="whitegrid", context="notebook", font_scale=1.1)
    ax = sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                     square=True, linewidths=.5, cbar_kws={'shrink': 0.7})
    plt.title('Correlation Matrix with Target', fontsize=16)
    plt.tight_layout()
    plt.show()

    print(f"\n상관관계 행렬 (including {target_name})")
    display(corr_matrix)


def analyze_correlation_with_target(X, y, top_n=15):
    """타겟 변수와의 상관관계 분석 (X, y 입력)"""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if hasattr(y, "name") and y.name is not None:
        target_name = y.name
    else:
        target_name = "Target"

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("수치형 특성이 없습니다.")
        return

    # y 값이 수치형 series/vector인가 확인
    y_array = y if isinstance(y, pd.Series) else pd.Series(y)
    correlations = {}
    for col in numeric_cols:
        correlations[col] = X[col].corr(y_array)

    corr_series = pd.Series(correlations).sort_values(key=abs, ascending=False)
    top_corr = corr_series.head(top_n)

    sns.set(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#e74c3c' if x > 0 else '#3498db' for x in top_corr.values]
    top_corr.plot(kind='barh', ax=ax, color=colors, edgecolor='black')
    ax.set_xlabel(f'Correlation with {target_name}')
    ax.set_title(f'Top {top_n} Variables Correlated with {target_name}')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.show()

    print(f"\n{target_name}와 상관관계가 높은 상위 {top_n}개 변수:")
    display(pd.DataFrame({
        'Correlation': top_corr,
        'Abs_Correlation': top_corr.abs()
    }).sort_values('Abs_Correlation', ascending=False))


def analyze_categorical_variables(df, max_categories=10):
    """범주형 변수 분포 분석"""
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not categorical_cols:
        print("범주형 변수가 없습니다.")
        return
    
    print("범주형 변수별 고유값 개수:")
    for col in categorical_cols:
        n_unique = df[col].nunique()
        print(f"  {col}: {n_unique}개 고유값")
        if n_unique <= max_categories:
            print(f"    분포: {df[col].value_counts().to_dict()}")
    
    # ODD_CP와의 관계가 있는 범주형 변수 분석
    print("\n범주형 변수별 ODD_CP 분포:")
    for col in categorical_cols:
        if df[col].nunique() <= max_categories:
            print(f"\n{col}:")
            crosstab = pd.crosstab(df[col], df['ODD_CP'], normalize='index') * 100
            display(crosstab)


def analyze_outliers(df, method='iqr'):
    """이상치 탐지 및 분석"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'ODD_CP' in numeric_cols:
        numeric_cols.remove('ODD_CP')
    
    outlier_summary = []
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        n_outliers = len(outliers)
        pct_outliers = (n_outliers / len(df)) * 100
        
        if n_outliers > 0:
            outlier_summary.append({
                'Variable': col,
                'Outlier_Count': n_outliers,
                'Outlier_Percentage': round(pct_outliers, 2),
                'Lower_Bound': round(lower_bound, 2),
                'Upper_Bound': round(upper_bound, 2)
            })
    
    if outlier_summary:
        outlier_df = pd.DataFrame(outlier_summary).sort_values('Outlier_Percentage', ascending=False)
        print("이상치가 있는 변수:")
        display(outlier_df)
    else:
        print("IQR 방법으로 탐지된 이상치가 없습니다.")