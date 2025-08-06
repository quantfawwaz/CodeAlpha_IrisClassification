import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Gandhi Ji\Downloads\Iris.csv") 

print("🔹 First 5 rows of the dataset:")
print(df.head())

print(f"\n🔹 Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

print("\n🔹 Dataset info:")
print(df.info())

print("\n🔹 Missing values in each column:")
print(df.isnull().sum())

print("\n🔹 Class distribution:")
print(df['Species'].value_counts())

sns.pairplot(df, hue='Species')
plt.title("Pairplot of Iris Dataset")
plt.show()


df.drop("Id", axis=1, inplace=True)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

print("\nEncoded Species column:")
print(df['Species'].value_counts())

X = df.drop('Species', axis=1)
y = df['Species']               

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('📌 Confusion Matrix for Iris Classification')
plt.show()