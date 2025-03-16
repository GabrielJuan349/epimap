import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class MortalityPredictionModel:
    """
    XGBoost model for predicting mortality rates.
    """
    
    def __init__(self, age_columns: list, target_column: str='Death1', **xgb_params):
        """
        Initialize the mortality prediction model.
        
        Parameters:
        -----------
        age_columns : list
            List of column names containing death counts by age range
        target_column : str
            Name of the column to predict (default: 'Death1')
        xgb_params : dict
            XGBoost model parameters
        """
        self.age_columns = age_columns
        self.target_column = target_column
        self.model = None
        self.features = None
        
        # Default XGBoost parameters
        self.xgb_params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        # Update with user-provided parameters
        self.xgb_params.update(xgb_params)
    
    def train(self, data, min_year, max_year):
        """
        Train the XGBoost model using time-based split.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data containing mortality information
        min_year : int
            Minimum year for training data
        max_year : int
            Maximum year for data (test set will include max_year-1 and max_year)
            
        Returns:
        --------
        dict
            Training results including model and evaluation metrics
        """
        # Time-based split
        train_years = range(min_year, max_year-1)
        test_years = [max_year-1, max_year]
        
        train_data = data[data['Year'].isin(train_years)]
        test_data = data[data['Year'].isin(test_years)]
        
        # Define features - exclude target and identifiers
        self.features = [col for col in data.columns if col not in [self.target_column] + self.age_columns]
        
        # Prepare training and testing data
        X_train = train_data[self.features]
        y_train = train_data[self.target_column]
        X_test = test_data[self.features]
        y_test = test_data[self.target_column]
        
        # Initialize and train model
        self.model = xgb.XGBRegressor(**self.xgb_params)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train, y_pred_train)
        test_metrics = self._calculate_metrics(y_test, y_pred_test)
        
        # Store results
        results = {
            'model': self.model,
            'features': self.features,
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
        
        return results
    
    def predict(self, data):
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data to make predictions on
            
        Returns:
        --------
        numpy.ndarray
            Predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Handle categorical features
        if 'Country' in data.columns and 'Country' not in self.features:
            data = pd.get_dummies(data, columns=['Country'], drop_first=True)
            
            # TODO: Check this code section.
            # Add missing columns if any
            for feature in self.features:
                if feature not in data.columns and feature.startswith('Country_'):
                    data[feature] = 0
        
        # Select relevant features
        X = data[self.features]
        return self.model.predict(X)
    
    def _calculate_metrics(self, y_true, y_pred):
        """
        Calculate evaluation metrics.
        
        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
            
        Returns:
        --------
        dict
            Dictionary of metrics
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }


def evaluate_model(model_results, model_name="XGBoost Mortality Model"):
    """
    Evaluate model performance and print metrics.
    
    Parameters:
    -----------
    model_results : dict
        Results from model training
    model_name : str
        Name of the model for reporting
        
    Returns:
    --------
    dict
        Evaluation results
    """
    train_metrics = model_results['train_metrics']
    test_metrics = model_results['test_metrics']
    
    print(f"\n=== Model Evaluation: {model_name} ===")
    print("\nTraining Metrics:")
    print(f"RMSE: {train_metrics['rmse']:.4f}")
    print(f"MAE: {train_metrics['mae']:.4f}")
    print(f"R²: {train_metrics['r2']:.4f}")
    
    print("\nTest Metrics:")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"MAE: {test_metrics['mae']:.4f}")
    print(f"R²: {test_metrics['r2']:.4f}")
    
    # Calculate prediction error distribution
    y_test = model_results['y_test']
    y_pred = model_results['y_pred_test']
    errors = y_test - y_pred
    
    error_stats = {
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'std_error': np.std(errors),
        'max_overprediction': np.min(errors),
        'max_underprediction': np.max(errors)
    }
    
    print("\nError Distribution:")
    print(f"Mean Error: {error_stats['mean_error']:.4f}")
    print(f"Median Error: {error_stats['median_error']:.4f}")
    print(f"Std Dev of Error: {error_stats['std_error']:.4f}")
    print(f"Max Overprediction: {error_stats['max_overprediction']:.4f}")
    print(f"Max Underprediction: {error_stats['max_underprediction']:.4f}")
    
    return {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'error_stats': error_stats
    }


def visualize_results(model_results, model_name="XGBoost Mortality Model"):
    """
    Visualize model results with multiple plots.
    
    Parameters:
    -----------
    model_results : dict
        Results from model training
    model_name : str
        Name of the model for plot titles
    """
    y_test = model_results['y_test']
    y_pred = model_results['y_pred_test']
    model = model_results['model']
    features = model_results['features']
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Actual vs. Predicted Values
    ax1 = fig.add_subplot(221)
    ax1.scatter(y_test, y_pred, alpha=0.5)
    
    # Add ideal prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Actual vs. Predicted Values')
    ax1.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(y_test, y_pred)[0, 1]
    ax1.annotate(f'Correlation: {correlation:.4f}', 
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # 2. Error Distribution
    ax2 = fig.add_subplot(222)
    errors = y_test - y_pred
    sns.histplot(errors, kde=True, ax=ax2)
    ax2.set_xlabel('Prediction Error')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Prediction Errors')
    
    # Add error stats
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    ax2.axvline(mean_error, color='r', linestyle='--')
    ax2.axvline(mean_error + std_error, color='g', linestyle='--')
    ax2.axvline(mean_error - std_error, color='g', linestyle='--')
    
    # Add annotation
    ax2.annotate(f'Mean: {mean_error:.4f}\nStd Dev: {std_error:.4f}', 
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # 3. Feature Importance
    ax3 = fig.add_subplot(223)
    
    # Get feature importance
    importance = model.feature_importances_
    indices = np.argsort(importance)[-10:]  # Top 10 features
    
    ax3.barh(range(len(indices)), importance[indices])
    ax3.set_yticks(range(len(indices)))
    ax3.set_yticklabels([features[i] for i in indices])
    ax3.set_xlabel('Importance')
    ax3.set_title('Top 10 Feature Importance')
    
    # 4. Residuals vs. Predicted Values
    ax4 = fig.add_subplot(224)
    ax4.scatter(y_pred, errors, alpha=0.5)
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_xlabel('Predicted Values')
    ax4.set_ylabel('Residuals')
    ax4.set_title('Residuals vs. Predicted Values')
    ax4.grid(True, alpha=0.3)
    
    # Add residual stats
    ax4.annotate(f'Residual Mean: {mean_error:.4f}', 
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle(f"Model Evaluation: {model_name}", fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()


def get_prediction_by_ids(results, country_id, disease_id):
    """
    Get prediction for a specific country and disease.
    """
    # Get the test data
    X_test = results['X_test']
    y_test = results['y_test']
    y_pred = results['y_pred_test']
    
    # Find the index where country_id and disease_id match
    print(X_test)
    mask = (X_test['Country'] == country_id) & (X_test['Cause_Code'] == disease_id)
    
    if not mask.any():
        return {"error": f"No data found for country ID {country_id} and disease ID {disease_id}"}
    
    # Get the first matching index
    idx = mask.idxmax()
    
    # Get prediction and actual value
    prediction = y_pred[X_test.index.get_loc(idx)] * 100
    actual = y_test.iloc[X_test.index.get_loc(idx)] * 100
    
    return {
        "country_id": country_id,
        "disease_id": disease_id,
        "predicted": f"{prediction:.2f}%",
        "actual": f"{actual:.2f}%"
    }

def get_prediction(data: pd.DataFrame, age_columns: list) -> pd.DataFrame:
    """
    Predict mortality for each age range in age_columns.


    pandas.DataFrame
        DataFrame containing the predicted mortality for each age range
    """
    # Create a copy of data to avoid modifying the original
    result_df = data.copy()
    
    # Create a dictionary to store predictions for each age range
    predictions = {}
    
    # For each age column, train a model and make predictions
    for target_column in age_columns:
        # Create a model for this age range
        model = MortalityPredictionModel(
            age_columns=[col for col in age_columns if col != target_column],
            target_column=target_column
        )
        
        # Train the model (using last year as test)
        min_year = data['Year'].min()
        max_year = data['Year'].max()
        results = model.train(data, min_year=min_year, max_year=max_year)
        
        # Generate predictions for the entire dataset
        X_predict = data[results['features']]
        predictions[target_column] = model.model.predict(X_predict)
    
    # Add predictions to the result DataFrame
    for column, preds in predictions.items():
        result_df[f'{column}_predicted'] = np.maximum(preds * 100, 0)
    
    return result_df