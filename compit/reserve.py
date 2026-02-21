"""Classification pipeline for competition submission using DecisionTreeClassifier."""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score

# Загрузка данных
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Предобработка данных
# Удаление строк с пропущенными метками
train_df = train_df.dropna(subset=['label'])

# Преобразование меток в целые числа
train_df['label'] = train_df['label'].astype(int)

# Разделение на признаки и метки
X = train_df.drop(columns=['label', 'id'])
y = train_df['label']

# Проверка количества классов
print("Уникальные метки в данных:", np.unique(y))

# Разделение на тренировочную и валидационную выборки
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


clf = RandomForestClassifier(
    n_estimators=500,
    min_samples_split=5,
    min_samples_leaf=1,
    max_features='sqrt',
    max_depth=20,
    criterion='gini',
    bootstrap=False,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    max_leaf_nodes=100,
    oob_score=True,
)
clf.fit(X_train, y_train)
print(f"OOB score: {clf.oob_score_:.4f}")

# Предсказание вероятностей для валидационной выборки
y_pred_proba = clf.predict_proba(X_val)

# Расчет метрик
val_auroc = roc_auc_score(
    y_val,
    y_pred_proba,
    multi_class='ovo',
    average='macro'
)
print(f"\nValidation AUROC: {val_auroc:.4f}")

# Кросс-валидация
cv_scores = cross_val_score(
    clf,
    X,
    y,
    cv=5,
    scoring='roc_auc_ovo',
    n_jobs=-1
)
print(f"Средний AUROC (кросс-валидация): {cv_scores.mean():.4f}")

# Обработка тестовых данных
test_features = test_df.drop(columns=['id'])

# Проверка наличия меток для тестирования
if 'label' in test_df.columns:
    y_test = test_df['label'].astype(int)
    test_pred_proba = clf.predict_proba(test_features)
    test_auroc = roc_auc_score(
        y_test,
        test_pred_proba,
        multi_class='ovo',
        average='macro'
    )
    print(f"\nTest AUROC: {test_auroc:.4f}")

# Генерация файла предсказаний
submission = pd.DataFrame({
    'id': test_df['id'],
    'label': clf.predict(test_features)
})

# Сохранение результатов
submission.to_csv("Tadjiev_Georgiy.csv", index=False)
print("\nФайл успешно создан.")
