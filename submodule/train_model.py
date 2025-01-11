from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def preprocess_data(data):
    """Preprocess the data by scaling and handling class imbalance."""
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    
    smote = SMOTE()
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    """Train a model and evaluate its performance."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    return score * 100, report, cm

def train_models(models, model_names, data):
    """Train and evaluate multiple models."""
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    scores, reports, cms = [], {}, {}
    for model, name in zip(models, model_names):
        score, report, cm = train_and_evaluate_model(model, X_train, X_test, y_train, y_test)
        scores.append(score)
        reports[name] = report
        cms[name] = cm
    
    return scores, reports, cms
