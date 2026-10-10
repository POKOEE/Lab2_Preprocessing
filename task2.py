import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ------------------------------------------------------------
# 1. Загрузка данных
# ------------------------------------------------------------
print("=" * 60)
print("ЗАДАНИЕ 2. ЗАГРУЗКА ДАННЫХ")
print("=" * 60)

try:
    df2 = sns.load_dataset('titanic')
except Exception:
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    df2 = pd.read_csv(url)

print(f"Размер датасета: {df2.shape}")
print("\nПервые 5 строк:")
print(df2.head())

print("\nПропуски в столбцах:")
print(df2.isnull().sum())

# ------------------------------------------------------------
# 2. Заполнение пропусков СРЕДНИМ (в отличие от Задания 1)
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("ШАГ 2. ЗАПОЛНЕНИЕ ПРОПУСКОВ СРЕДНИМ ЗНАЧЕНИЕМ")
print("=" * 60)

# age — заполняем СРЕДНИМ (а не медианой)
mean_age = df2['age'].mean()
df2['age'] = df2['age'].fillna(mean_age)
print(f"Среднее значение age = {mean_age:.4f}")
print(f"Медиана age для сравнения = {df2['age'].median():.4f}")

# embarked — заполняем модой
mode_embarked = df2['embarked'].mode()[0]
df2['embarked'] = df2['embarked'].fillna(mode_embarked)
print(f"embarked заполнен модой = {mode_embarked}")

# Удаление столбцов
df2.drop(['deck', 'embark_town'], axis=1, inplace=True)
print("Столбцы 'deck' и 'embark_town' удалены")

print("\nПропуски после обработки:")
print(df2.isnull().sum())

# ------------------------------------------------------------
# 3. Кодирование категориальных признаков
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("ШАГ 3. КОДИРОВАНИЕ ПРИЗНАКОВ")
print("=" * 60)

# Статистика ДО масштабирования
print("\nСтатистика ДО масштабирования (age, fare):")
print(df2[['age', 'fare']].describe())

# Label Encoding для sex
le = LabelEncoder()
df2['sex_encoded'] = le.fit_transform(df2['sex'])
print(f"\nLabel Encoding для 'sex':")
print(f"Классы: {le.classes_}")
print(f"Кодировка: {list(zip(le.classes_, le.transform(le.classes_)))}")

# One-Hot Encoding для embarked
df2 = pd.get_dummies(df2, columns=['embarked'], drop_first=True)

# Удаляем лишние столбцы и исходный 'sex' (теперь есть sex_encoded)
df2.drop(['pclass', 'who', 'adult_male', 'class', 'alive', 'alone', 'sex'],
         axis=1, inplace=True)

print("\nДатасет после кодирования (первые 5 строк):")
print(df2.head())

print(f"\nСтолбцы после кодирования: {list(df2.columns)}")

# ------------------------------------------------------------
# 4. Нормализация MinMaxScaler (вместо StandardScaler)
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("ШАГ 4. НОРМАЛИЗАЦИЯ (MinMaxScaler)")
print("=" * 60)

scaler = MinMaxScaler()
cols_to_scale = ['age', 'fare']
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])

print("\nДатасет после нормализации (первые 5 строк):")
print(df2.head())

print("\nСтатистика ПОСЛЕ масштабирования (age, fare):")
print(df2[['age', 'fare']].describe())

# ------------------------------------------------------------
# 5. Сравнение статистик ДО и ПОСЛЕ
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("СРАВНЕНИЕ СТАТИСТИК ДО и ПОСЛЕ МАСШТАБИРОВАНИЯ")
print("=" * 60)

print("\nЧто изменилось:")
print("- mean (среднее) сдвинулось к ~0.3-0.4 (не 0, как в StandardScaler)")
print("- min стало ровно 0 (было 0.42 для age и 0 для fare)")
print("- max стало ровно 1 (было 80 для age и 512.33 для fare)")
print("- Все значения теперь в диапазоне [0, 1]")

# ------------------------------------------------------------
# 6. Визуализация распределений до/после
# ------------------------------------------------------------
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.hist(df2['age'], bins=30, color='steelblue', edgecolor='black')
plt.title('age после MinMaxScaler')
plt.xlabel('age (нормализованное)')

plt.subplot(2, 2, 2)
plt.hist(df2['fare'], bins=30, color='orange', edgecolor='black')
plt.title('fare после MinMaxScaler')
plt.xlabel('fare (нормализованное)')

plt.subplot(2, 2, 3)
plt.boxplot(df2['age'])
plt.title('Boxplot age (после)')

plt.subplot(2, 2, 4)
plt.boxplot(df2['fare'])
plt.title('Boxplot fare (после)')

plt.tight_layout()
plt.savefig('task2_distributions.png', dpi=150)
plt.show()

print("\nГрафики сохранены в файл 'task2_distributions.png'")

# ------------------------------------------------------------
# 7. Анализ Label Encoding
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("АНАЛИЗ ПРИМЕНЕНИЯ LABEL ENCODING ДЛЯ 'sex'")
print("=" * 60)

print("\nLabel Encoding для признака 'sex' может быть некорректным, потому что:")
print("1. Модель может ошибочно интерпретировать коды как порядковые (0 < 1),")
print("   хотя между 'female' и 'male' нет иерархии.")
print("2. Это может привести к неверным весовым коэффициентам в линейных моделях:")
print("   модель решит, что 'male' в 2 раза 'больше' female, что бессмысленно.")
print("3. Для бинарных признаков без естественного порядка лучше использовать")
print("   One-Hot Encoding — тогда модель обучит отдельный вес для каждого пола.")
print("\nКОГДА LABEL ENCODING ДОПУСТИМ:")
print("- Для деревьев решений (Random Forest, Gradient Boosting) — им не важен")
print("  порядок кодов, они строят разбиения по порогам.")
print("- Для бинарных признаков, если модель нелинейная.")
print("- Для порядковых признаков (например, 'маленький/средний/большой').")
print("\nКОГДА НЕДОПУСТИМ:")
print("- Для линейных моделей (логистическая регрессия, SVM, kNN).")
print("- Для номинальных признаков с 3+ категориями (там точно нет порядка).")

# ------------------------------------------------------------
# 8. Финальный вид датасета
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("ФИНАЛЬНЫЙ ДАТАСЕТ (Задание 2):")
print("=" * 60)
print(df2.head())
print(f"\nРазмер итогового датасета: {df2.shape}")