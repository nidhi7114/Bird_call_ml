import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
import xgboost as xgb

def load_data(file_path):
    #Load the dataset from an Excel file.
    return pd.read_excel(file_path)


def preprocess_data(df):
    """
    Preprocess the data by separating features and labels, and splitting into train and test sets.
    tuple: Train-test split data (X_train, X_test, y_train, y_test)
    """
    # Separate features and labels
    df_refined = df.dropna()
    X = df_refined.drop(columns=['filename', 'species'])
    y = df['species']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test


def standardize_data(X_train, X_test):
    """
    X_train (pd.DataFrame): Training feature data.
    X_test (pd.DataFrame): Testing feature data.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def train_knn_classifier(X_train_scaled, y_train, n_neighbors=3):
    """
    Train a k-Nearest Neighbors classifier.
    
    Args:
    X_train_scaled (pd.DataFrame): Scaled training feature data.
    y_train (pd.Series): Training labels.
    n_neighbors (int): Number of neighbors for k-NN.
    
    Returns:
    KNeighborsClassifier: Trained k-NN model.
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_scaled, y_train)
    return knn

# Train the Random Forest classifier
def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    #return the trained model 
    return rf
# train adaboost
def train_ada_boost(X_train, y_train):
    # Create a base model (weak learner)
    base_model = DecisionTreeClassifier(max_depth=1)

    # Create AdaBoost model
    ada_model = AdaBoostClassifier(estimator = base_model, n_estimators=150, random_state=0)

    # Train the model
    ada_model.fit(X_train, y_train)
    return ada_model

#train catboost classifier
def train_cat_boost(X_train, y_train):

    # Create CatBoost model
    catboost_model = CatBoostClassifier(iterations=100, learning_rate=0.1, random_state=42, verbose=0)

    # Train the model
    catboost_model.fit(X_train, y_train)
    return catboost_model

#train xgboost classifier
def train_xg_boost(X_train, y_train):

    # Create CatBoost model
    xg_boost_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

    # Train the model
    xg_boost_model.fit(X_train, y_train)
    return xg_boost_model


def evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test):

    #Evaluate the trained model and print results.
    
    # Predictions on training and testing sets
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred =model.predict(X_test_scaled)
    
    # Compute accuracies
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # compute F1 scores
    train_f1_score = f1_score(y_train,y_train_pred, average ='macro')
    test_f1_score = f1_score(y_test, y_test_pred, average = 'macro')
    
    # Confusion matrix and classification report for test data
    #conf_matrix = confusion_matrix(y_test, y_test_pred)
    #class_report = classification_report(y_test, y_test_pred, zero_division =1)
    
    print(f"Training Accuracy: {train_accuracy:.2f}")
    print(f"Testing Accuracy: {test_accuracy:.2f}")

    #print f1 scores
    print(f"Training F1 Score: {train_f1_score:.2f}")
    print(f"Testing F1 Score: {test_f1_score:.2f}")
    #print("Confusion Matrix:")
    #print(conf_matrix)
    #print("Classification Report:")
    #print(class_report)
    #return train_accuracy,test_accuracy,train_f1_score,test_f1_score

def knn_eval(X_train_scaled, X_test_scaled, y_train, y_test,neighbours = 3):
    # Train model
    # Initialize the k-NN classifier and train it 
    knn = train_knn_classifier(X_train_scaled,y_train,3)
    #evaluate the model on test data and return the accuracies and F1 score
    print("k-NN")
    evaluate_model(knn, X_train_scaled,X_test_scaled,y_train,y_test)

def random_forests_eval(X_train_scaled, X_test_scaled, y_train, y_test):
    # Train model
    # Initialize the k-NN classifier and train it 
    rf = train_random_forest(X_train_scaled,y_train)
    #evaluate the model on test data and return the accuracies and F1 score
    print("Random Forests")
    evaluate_model(rf, X_train_scaled,X_test_scaled,y_train,y_test)

def ada_boost_eval(X_train_scaled, X_test_scaled, y_train, y_test):
    # Train model
    # Initialize the k-NN classifier and train it 
    ada_model = train_ada_boost(X_train_scaled,y_train)
    #evaluate the model on test data and return the accuracies and F1 score
    print("Ada Boost")
    evaluate_model(ada_model, X_train_scaled,X_test_scaled,y_train,y_test)

def cat_boost_eval(X_train_scaled, X_test_scaled, y_train, y_test):
    # Train model
    # Initialize the k-NN classifier and train it 
    cat_model = train_cat_boost(X_train_scaled,y_train)
    #evaluate the model on test data and return the accuracies and F1 score
    print("CatBoost")
    evaluate_model(cat_model, X_train_scaled,X_test_scaled,y_train,y_test)

def xg_boost_eval(X_train_scaled, X_test_scaled, y_train, y_test):
    # Train model
    # Initialize the k-NN classifier and train it 
    xg_boost_model = train_xg_boost(X_train_scaled,y_train)
    #evaluate the model on test data and return the accuracies and F1 score
    print("CatBoost")
    evaluate_model(cat_model, X_train_scaled,X_test_scaled,y_train,y_test)

def main():
    # Load and preprocess data
    file_path = 'C:\\Users\\NEW\\Documents\\Python Scripts\\bird_species_features.xlsx'
    df = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Standardize features
    X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)
    '''
    #evaluating k-NN
    knn_eval(X_train_scaled, X_test_scaled,y_train, y_test)
    #evaluating Random Forests
    random_forests_eval(X_train_scaled, X_test_scaled,y_train, y_test)
    #evaluating  Ada Boost
    ada_boost_eval(X_train_scaled, X_test_scaled,y_train, y_test)
    '''
    #evaluating catboost
    cat_boost_eval(X_train_scaled, X_test_scaled,y_train, y_test)

if __name__ == "__main__":
    main()
