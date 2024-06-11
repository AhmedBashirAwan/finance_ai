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
        # Comparing or Debt Paying
        if not request.values:
            raise HTTPException(status_code=400, detail="Values are required for comparing or debt paying")
        
        gets = request.values
        sumi = sum(gets)
        isumi = sum([models[target].predict(preddf).item() for target in targets])
        
        result = {"message": "Your current budgeting is good." if sumi <= isumi else "Your current budgeting can be improved."}

        if request.choice == 3:
            if request.debt is None:
                raise HTTPException(status_code=400, detail="Debt is required for debt paying choice")
            debt = request.debt
            v1 = debt / (bi - sumi)
            v2 = debt / (bi - isumi)
            result.update({
                "current_budgeting_months": round(v1),
                "average_budgeting_months": round(v2)
            })

        return result

    else:
        raise HTTPException(status_code=400, detail="Invalid choice")

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
