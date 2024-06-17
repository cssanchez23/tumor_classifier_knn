"""THis script is used to used KNN to determine if a tumor is malignant or benign using nearest neighbore"""
import pandas as pd
import scipy.stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier



wbcd = pd.read_csv("~/Desktop/code/code/wisc_bc_data.csv")

## KNN practice 
wbcd.drop(columns=wbcd.columns[0], axis=1, inplace=True)
wbcd["diagnosis"].hist(bins=2)
# Ensure that 'diagnosis' column is a factor
wbcd['diagnosis'] = wbcd['diagnosis'].astype('category')
wbcd['diagnosis'] = wbcd['diagnosis'].cat.rename_categories({'B': 'Benign', 'M': 'Malignant'})

# Normalize the features
scaler = MinMaxScaler()  # python has built in normalizer
#scaler = StandardScaler()   this is how to scale with z score 
wbcd_n = pd.DataFrame(scaler.fit_transform(wbcd.iloc[:, 1:]))

# Split the data into train and test sets
wbcd_train = wbcd_n.iloc[:469, :] # everything normalized but the first column
wbcd_test = wbcd_n.iloc[469:569, :]
# Store class labels in factor vectors
wbcd_train_labels = wbcd.iloc[:469, 0] # this has just the first column
wbcd_test_labels = wbcd.iloc[469:569, 0]

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(wbcd_train, wbcd_train_labels)
# Make predictions on the test set
wbcd_test_pred = knn.predict(wbcd_test)
cross_table = pd.crosstab(index=wbcd_test_labels, columns=wbcd_test_pred, rownames=['Actual'], colnames=['Predicted'])
cross_table
