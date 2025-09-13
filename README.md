# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sklearn.linear_model import LinearRegression
import numpy as np
import joblib
import datetime

app = FastAPI()

# Dummy database
expenses = []
goals = []

# Load or create model
try:
    model = joblib.load("spending_model.pkl")
except:
    model = LinearRegression()
    model.fit(np.array([[0]]), np.array([0]))

# Models
class Expense(BaseModel):
    date: str
    amount: float
    category: str
    description: Optional[str]

class Goal(BaseModel):
    name: str
    target_amount: float
    saved_amount: float

# Routes
@app.post("/add_expense/")
def add_expense(expense: Expense):
    expenses.append(expense.dict())
    return {"message": "Expense added successfully"}

@app.get("/get_expenses/")
def get_expenses():
    return expenses

@app.post("/add_goal/")
def add_goal(goal: Goal):
    goals.append(goal.dict())
    return {"message": "Goal added successfully"}

@app.get("/get_goals/")
def get_goals():
    return goals

@app.get("/predict_spending/")
def predict_spending(days: int = 30):
    if not expenses:
        raise HTTPException(status_code=404, detail="No expenses found")
    dates = [datetime.datetime.strptime(e['date'], "%Y-%m-%d").timestamp() for e in expenses]
    amounts = [e['amount'] for e in expenses]
    X = np.array(dates).reshape(-1, 1)
    y = np.array(amounts)
    model.fit(X, y)
    future_date = datetime.datetime.now() + datetime.timedelta(days=days)
    pred = model.predict([[future_date.timestamp()]])
    joblib.dump(model, "spending_model.pkl")
    return {"predicted_spending": round(float(pred[0]), 2)}

@app.get("/budget_alert/")
def budget_alert(threshold: float):
    total = sum(e['amount'] for e in expenses)
    alert = total > threshold
    return {"total_spent": total, "alert": alert}
