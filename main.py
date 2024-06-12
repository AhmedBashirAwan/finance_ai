import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

# Initialize FastAPI
app = FastAPI()

# Load the data
df_from_csv = pd.read_csv('generated_data.csv', encoding='utf-8')

# Calculate Remainder
df_from_csv['Remainder'] = df_from_csv['Basic Income'] - (
    df_from_csv[" Rent"] + df_from_csv[' Food'] + df_from_csv[' Transportation'] +
    df_from_csv[' Utilities'] + df_from_csv[' Clothing'] + df_from_csv[' Leisure'] +
    df_from_csv[' Healthcare']
)

df = df_from_csv

# Define the KNN Regression function
def knn_regression(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the KNN Regressor
    knn = KNeighborsRegressor(n_neighbors=5)
    
    # Train the model
    knn.fit(X_train, y_train)
    
    # Make predictions
    y_pred = knn.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print the metrics
    print("Mean Absolute Error (MAE):", mae)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Absolute Percentage Error (MAPE):", mape)
    print("R-squared score:", r2)
    print("--" * 20)
    
    return knn

# Define the targets
targets = [' Rent', ' Food', ' Transportation', ' Utilities', ' Clothing', ' Leisure', ' Healthcare']

# Train the model for each target
models = {}
for target in targets:
    X = df[['Basic Income', ' People']]
    y = df[target]
    print(f"Training KNN Regression for {target}")
    models[target] = knn_regression(X, y)

# Define the request model
class BudgetRequest(BaseModel):
    basic_income: float
    number_of_people: float
    values: List[int] = None
    debt: float = None
    choice: int

# Define the POST endpoint
@app.post("/budget/")
def create_budget(request: BudgetRequest):
    bi = request.basic_income
    peeps = request.number_of_people
    preddf = pd.DataFrame({'Basic Income': [bi], ' People': [peeps]})

    if request.choice == 1:
        # Prediction
        pred_values = {}
        for target in targets:
            pred_values[target] = models[target].predict(preddf).item()
        return {"predicted_values": pred_values}

    elif request.choice in [2, 3]:
        # Ensure predy and gets are populated
        if request.values is None or len(request.values) != len(targets):
            raise HTTPException(status_code=400, detail="Values must be provided for all targets.")
        
        predy = [models[target].predict(preddf).item() for target in targets]
        gets = request.values

        if request.choice == 2:
            # Comparing
            comparison_results = []
            for i in range(len(targets)):
                diff = predy[i] * 0.2
                if abs(predy[i] - gets[i]) > abs(diff):
                    comparison_results.append(f"{targets[i]} is taking too much, restrain it to around {predy[i]}")
            return {"comparison_results": comparison_results}

        elif request.choice == 3:
            # Debt Paying
            sumi = sum(gets)
            isumi = sum(predy)
            debt = request.debt

            debt_results = []
            if sumi <= isumi:
                remaining_budget = bi - sumi
                if remaining_budget <= 0:
                    debt_results.append(f"With your current budgeting you can pay: {remaining_budget}, so you will have to increase income")
                else:
                    debt_results.append(f"Your current budgeting is good you can pay: {remaining_budget}, but if you budget according to average you can pay: {bi - isumi}")
            else:
                remaining_budget = bi - sumi
                remaining_average_budget = bi - isumi
                if remaining_budget <= 0:
                    if remaining_average_budget <= 0:
                        debt_results.append(f"With your current budgeting you can pay: {remaining_budget}, and even if you budget according to average you can pay: {(remaining_average_budget)}, so you will have to increase income")
                    else:
                        debt_results.append(f"With your current budgeting you can pay: {remaining_budget}, but if you budget according to average you can pay: {remaining_average_budget}")
                else:
                    debt_results.append(f"Your current budgeting allows you to pay: {remaining_budget}, but it can be improved to: {remaining_average_budget}")

            if debt is not None:
                v1 = debt / (bi - sumi)
                v2 = debt / (bi - isumi)
                if v1 <= 0:
                    debt_results.append("You can't pay debt like this.")
                    if v2 <= 0:
                        debt_results.append("You can't pay debt even according to average budgeting")
                    elif v2 < 1:
                        debt_results.append("If you budget according to average you can pay in around 1 month")
                    else:
                        debt_results.append(f"If you budget according to average you can pay in around {round(v2)} months")
                elif v1 < 1:
                    debt_results.append("You can pay debt in around 1 month")
                    if v2 <= 0:
                        debt_results.append("You can't pay debt in ideal budgeting")
                    elif v2 < 1:
                        debt_results.append("If you budget according to average you can pay in around 1 month")
                    else:
                        debt_results.append(f"If you budget according to average you can pay in around {round(v2)} months")
                else:
                    debt_results.append(f"You can pay in around {round(v1)} months")
                    if v2 <= 0:
                        debt_results.append("You can't pay debt in ideal budgeting")
                    elif v2 < 1:
                        debt_results.append("If you budget according to average you can pay in around 1 month")
                    else:
                        debt_results.append(f"If you budget according to average you can pay in around {round(v2)} months")

            return {"debt_results": debt_results}
    
    else:
        raise HTTPException(status_code=400, detail="Invalid choice")


# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)