from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Model(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass
    
    @abstractmethod
    def predict(self, X_test):
        pass

class LinearRegressionModel(Model):
    def __init__(self):
        self.model = LinearRegression()
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)

class RandomForestModel(Model):
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)

class EvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(self, y_true, y_pred):
        pass

class MSEStrategy(EvaluationStrategy):
    def evaluate(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

class RMSEStrategy(EvaluationStrategy):
    def evaluate(self, y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

class R2Strategy(EvaluationStrategy):
    def evaluate(self, y_true, y_pred):
        return r2_score(y_true, y_pred)

class ModelEvaluator:
    def __init__(self):
        self.strategies = {
            'MSE': MSEStrategy(),
            'RMSE': RMSEStrategy(),
            'R2': R2Strategy()
        }
    
    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        results = {}
        for metric_name, strategy in self.strategies.items():
            results[metric_name] = strategy.evaluate(y_test, y_pred)
        return results