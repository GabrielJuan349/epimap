import argparse
import pandas as pd
import time

from model import MortalityPredictionModel, evaluate_model, visualize_results, get_prediction_by_ids, get_prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser("XGBoost trainer script")
    parser.add_argument("-d", "--data_path", help="Path to the data CSV file.", type=str, required=True)
    parser.add_argument("-t", "--target_column", help="Target column to predict", type=str, default="Deaths17-18")
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
    age_columns = ['Deaths2-6', 'Deaths7-8', 'Deaths9-10', 'Deaths11-12', 'Deaths13-14', 'Deaths15-16', 'Deaths17-18', 'Deaths19-20', 'Deaths21-22', 'Deaths23-24', 'Deaths25-26']
    # model = MortalityPredictionModel(age_columns=age_columns, target_column=target_column)

    # Train model
    # results = model.train(data, min_year=2000, max_year=2020)
    #vprint(get_prediction_by_ids(results, 1125, 1008))

    # Evaluate model
    #model_name = args.model_name
    #evaluation = evaluate_model(results, model_name=model_name)

    # Visualize results
    # visualize_results(results, model_name=model_name)
    start = time.time()
    preds = get_prediction(data, age_columns)
    print(f"Elapsed time {time.time() - start}")
    print("-"*50)

    # Filter for Country 5198 and Disease ID 1092
    disease_prediction = preds[(preds['Country'] == 5198) & (preds['Cause_Code'] == 1094)]

    # Extract the prediction for the specific age range
    if not disease_prediction.empty:
        prediction_value = disease_prediction['Deaths2-6_predicted'].values[0]
        # Assuming the original column is just called 'Deaths17-18' without '_actual' suffix
        actual_value = disease_prediction['Deaths2-6'].values[0]
        print(f"Country ID: 5198, Disease ID: 1092, Age Range: Deaths2-6")
        print(f"Predicted: {prediction_value:.4f}")
        print(f"Actual: {actual_value:.4f}")
    else:
        print(f"No data found for Disease ID 1092 in Country 5198")


