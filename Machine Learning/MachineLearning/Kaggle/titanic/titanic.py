# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# Import the dataset
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

# processing the data
# Sex
lb_enc_sex = LabelEncoder()
lb_sex = lb_enc_sex.fit_transform(data_train['Sex'])
oh_enc_sex = OneHotEncoder()
oh_enc_sex.fit(np.array(lb_sex).reshape(-1, 1))
# Age
age = data_train['Age'].fillna(data_train['Age'].mean())
# Fare
fare = data_train['Fare'].fillna(data_train['Fare'].mean())
# Embarked
data_train['Embarked'] = data_train['Embarked'].fillna('U')
lb_enc_emb = LabelEncoder()
lb_emb = lb_enc_emb.fit_transform(data_train['Embarked'])
oh_enc_emb = OneHotEncoder()
oh_enc_emb.fit(np.array(lb_emb).reshape(-1, 1))

def transform_data(df):

    # Sex
    lb_sex = lb_enc_sex.transform(df['Sex'])
    enc_sex = oh_enc_sex.transform(np.array(lb_sex).reshape(-1, 1))
    df_sex = pd.DataFrame(enc_sex.toarray(), columns=['male', 'female'])

    # Age
    df_age = pd.DataFrame(age, columns=['Age'])

    # Fare
    df_fare = pd.DataFrame(fare, columns=['Fare'])

    # Embarked
    lb_emb = lb_enc_emb.transform(df['Embarked'])
    enc_emb = oh_enc_emb.transform(np.array(lb_emb).reshape(-1, 1))
    df_emb = pd.DataFrame(enc_emb.toarray(), columns=['C', 'Q', 'S', 'U'])

    return pd.concat([df['Pclass'], df_sex, df['SibSp'], df['Parch'], df_fare, df_age, df_emb],axis=1)

X_train = transform_data(data_train)
y_train = data_train['Survived']

X_train = X_train.dropna()
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

X_test = transform_data(data_test)
X_test = sc.fit_transform(X_test)
X_test = pd.DataFrame(X_test).dropna()

"""
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)
explained_varience = pca.explained_variance_ratio_
"""

# Fitting SVC
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(y_pred)
print(y_pred.shape)

# Exporting to csv
PassengerId = np.array(data_test["PassengerId"]).astype(int)
predicted_data = pd.DataFrame(y_pred, PassengerId, columns=["Survived"])
predicted_data.to_csv("titanic.csv", index_label=["PassengerId"])

"""
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                               stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1,
                               stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.
             predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('SVM (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
"""