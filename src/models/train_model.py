import argparse
import pandas as pd

from model import MortalityPredictionModel, evaluate_model, visualize_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser("XGBoost trainer script")
    parser.add_argument("-d", "--data_path", help="Path to the data CSV file.", type=str, required=True)
    parser.add_argument("-t", "--target_column", help="Target column to predict", type=str, default="Deaths5-6")
    parser.add_argument("--min_age", help="Minimum age to use for prediction", type=str, default=2018)
    parser.add_argument("--max_age", help="Maximum age to use for prediction", type=str, default=2022)
    parser.add_argument("--model_name", help="Name the model", type=str, default="Global Mortality Model")
    args = parser.parse_args()

    # Reading data
    data_path = args.data_path
    data = pd.read_csv(data_path)

    # Training model
    min_age = args.min_age
    max_age = args.max_age
    target_column = args.target_column

    # Define the model
    age_columns = ['Deaths0-4', 'Deaths5-6', 'Deaths7-8', 'Deaths9-10', 'Deaths11-12', 'Deaths13-14', 'Deaths15-16', 'Deaths17-18', 'Deaths19-20', 'Deaths21-22', 'Deaths23-24', 'Deaths25-26', ]
    model = MortalityPredictionModel(age_columns=age_columns, target_column=target_column)

    # Train model
    results = model.train(data, min_year=2000, max_year=2020)

    # Evaluate model
    model_name = args.model_name
    # evaluation = evaluate_model(results, model_name=model_name)

    # Visualize results
    # visualize_results(results, model_name=model_name)
