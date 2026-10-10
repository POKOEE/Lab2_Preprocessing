import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# 1. Загрузка данных
# ------------------------------------------------------------
print("=" * 60)
print("ШАГ 1. ЗАГРУЗКА ДАННЫХ")
print("=" * 60)

try:
    df = sns.load_dataset('titanic')
except Exception:
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    df = pd.read_csv(url)

print(f"Размер датасета: {df.shape}")
print("\nИнформация о датасете:")
df.info()

print("\nПервые 5 строк датасета:")
print(df.head())

print("\nСтатистика по числовым признакам:")
print(df.describe())

# ------------------------------------------------------------
# 2. Обработка пропусков
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("ШАГ 2. ОБРАБОТКА ПРОПУСКОВ")
print("=" * 60)

print("\nКоличество пропусков в каждом столбце (ДО обработки):")
print(df.isnull().sum())

# age — заполняем медианой (устойчива к выбросам)
median_age = df['age'].median()
df['age'] = df['age'].fillna(median_age)
print(f"\nage заполнен медианой = {median_age:.2f}")

# embarked — заполняем модой
mode_embarked = df['embarked'].mode()[0]
df['embarked'] = df['embarked'].fillna(mode_embarked)
print(f"embarked заполнен модой = {mode_embarked}")

# deck и embark_town — удаляем
df.drop(['deck', 'embark_town'], axis=1, inplace=True)
print("Столбцы 'deck' и 'embark_town' удалены")

print("\nКоличество пропусков ПОСЛЕ обработки:")
print(df.isnull().sum())

# ------------------------------------------------------------
# 3. Кодирование категориальных признаков
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("ШАГ 3. КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ")
print("=" * 60)

# One-Hot Encoding для sex и embarked
df_processed = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True)

# Удаляем избыточные столбцы
df_processed.drop(['pclass', 'who', 'adult_male', 'class', 'alive', 'alone'],
                  axis=1, inplace=True)

print("\nДатасет после One-Hot Encoding (первые 5 строк):")
print(df_processed.head())

print(f"\nРазмер после кодирования: {df_processed.shape}")

# ------------------------------------------------------------
# 4. Масштабирование (стандартизация)
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("ШАГ 4. СТАНДАРТИЗАЦИЯ 'age' и 'fare'")
print("=" * 60)

scaler = StandardScaler()
cols_to_scale = ['age', 'fare']
df_processed[cols_to_scale] = scaler.fit_transform(df_processed[cols_to_scale])

print("\nДатасет после стандартизации (первые 5 строк):")
print(df_processed.head())

print("\nСтатистика после стандартизации:")
print(df_processed[cols_to_scale].describe())

# ------------------------------------------------------------
# 5. Работа с выбросами (метод IQR, capping)
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("ШАГ 5. ОБРАБОТКА ВЫБРОСОВ В 'fare' (метод IQR)")
print("=" * 60)

# Визуализация до обработки
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(y=df_processed['fare'])
plt.title('Boxplot для Fare (до обработки выбросов)')

# Расчёт границ IQR
Q1 = df_processed['fare'].quantile(0.25)
Q3 = df_processed['fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\nQ1 = {Q1:.3f}")
print(f"Q3 = {Q3:.3f}")
print(f"IQR = {IQR:.3f}")
print(f"Нижняя граница = {lower_bound:.3f}")
print(f"Верхняя граница = {upper_bound:.3f}")

# Capping — ограничение выбросов
df_processed['fare'] = np.where(df_processed['fare'] > upper_bound,
                                upper_bound, df_processed['fare'])
df_processed['fare'] = np.where(df_processed['fare'] < lower_bound,
                                lower_bound, df_processed['fare'])

# Визуализация после обработки
plt.subplot(1, 2, 2)
sns.boxplot(y=df_processed['fare'])
plt.title('Boxplot для Fare (после обработки выбросов)')

plt.tight_layout()
plt.savefig('fare_boxplot.png', dpi=150)
plt.show()

print("\nВыбросы ограничены границами (capping)")
print("График сохранён в файл 'fare_boxplot.png'")

# ------------------------------------------------------------
# 6. Итоговый результат
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("ФИНАЛЬНЫЙ ДАТАСЕТ (Задание 1):")
print("=" * 60)
print(df_processed.head())
print(f"\nРазмер итогового датасета: {df_processed.shape}")

# ------------------------------------------------------------
# 7. Ответ на вопрос задания 1
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("ОТВЕТ НА ВОПРОС:")
print("Почему для заполнения age использована медиана, а не среднее?")
print("=" * 60)
print("Возраст имеет скошенное распределение с выбросами.")
print("Среднее чувствительно к выбросам — они 'тянут' его в сторону.")
print("Медиана устойчива к выбросам и точнее отражает типичное значение.")
print("Поэтому для возраста медиана предпочтительнее среднего.")