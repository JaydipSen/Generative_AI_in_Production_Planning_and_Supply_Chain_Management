
# PPC Decision Intelligence Lab

## Generative AI in Production Planning and Supply Chain Management

An interactive Streamlit application designed for MBA teaching
and management education.

The application demonstrates how analytical tools can support
Production Planning and Control (PPC) decisions.

## Modules

### 1. Overview Dashboard
- Historical demand visualization
- Product master
- Material master
- Bill of Materials
- Work-centre information

### 2. Demand Forecasting
Forecasting methods:
- Moving Average
- Weighted Moving Average
- Exponential Smoothing
- Linear Trend

Performance measures:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

Visualizations:
- Historical demand
- Fitted forecast
- Future forecast
- Forecast errors

### 3. Material Requirements Planning (MRP)
- Product selection
- Production quantity input
- BOM explosion
- Gross requirements
- Net requirements
- Material shortages
- Procurement cost
- Supplier and lead-time information

Visualizations:
- Gross requirements versus available inventory
- Material shortage information

### 4. Inventory Management
- Economic Order Quantity (EOQ)
- Safety stock
- Reorder point
- Inventory days
- Service level assumptions
- ABC analysis

Visualizations:
- Available inventory versus reorder point
- ABC consumption-value chart

### 5. Aggregate Planning
Compare:
- Level production strategy
- Chase strategy
- Hybrid strategy

Cost components:
- Workforce cost
- Hiring cost
- Layoff cost
- Inventory carrying cost
- Shortage cost

Visualizations:
- Demand versus production
- Cost comparison
- Inventory evolution

### 6. Capacity Planning
- Work-centre workload
- Effective capacity
- Utilization
- Capacity overload
- Bottleneck identification

Visualizations:
- Workload versus capacity
- Work-centre utilization

### 7. Job-shop Scheduling
Scheduling rules:
- First Come First Served (FCFS)
- Shortest Processing Time (SPT)
- Earliest Due Date (EDD)

Performance indicators:
- Completion time
- Average tardiness
- Total tardiness

Visualizations:
- Production Gantt chart
- Due date versus completion comparison

### 8. AI Decision Support
Rule-based managerial interpretation of:
- Forecast accuracy
- Material shortages
- Inventory replenishment risk
- Capacity bottlenecks
- Production scheduling decisions

The current application does not require an external LLM API.

---

## Installation

### Step 1: Install Python

Install Python 3.10 or later.

Download from:

https://www.python.org/downloads/

### Step 2: Create a project folder

Create a folder named:

`PPC_Decision_Intelligence_Lab`

Place the following files inside it:

- app.py
- requirements.txt
- README.md

### Step 3: Install dependencies

Open Terminal or Command Prompt in the project directory.

Run:

```bash
pip install -r requirements.txt
```

### Step 4: Start Streamlit

Run:

```bash
streamlit run app.py
```

The application should open in the browser.

Default local address:

http://localhost:8501

---

## Dataset

The application uses synthetic illustrative manufacturing data.

The dataset includes:

- 24 months of historical demand
- Four manufactured products
- Five raw materials
- Bill of Materials
- Inventory information
- Supplier lead times
- Work-centre capacity
- Twelve illustrative production jobs

The data is generated inside the application, so no external
dataset or database is required.

---

## Classroom Use

Suggested classroom workflow:

1. Explore historical demand.
2. Compare forecasting methods.
3. Evaluate forecast errors.
4. Select a production quantity.
5. Explode the Bill of Materials.
6. Identify material shortages.
7. Calculate inventory policies.
8. Compare aggregate production plans.
9. Identify capacity bottlenecks.
10. Compare job-shop scheduling rules.
11. Interpret the results from a managerial perspective.

---

## Limitations

This is an educational prototype.

It is not a complete production planning system or ERP.

The application currently uses:

- Synthetic data
- Simplified BOM calculations
- Illustrative demand variability
- Simplified aggregate planning cost models
- Single-machine scheduling assumptions
- Simplified capacity requirements
- Rule-based interpretation rather than a live generative AI model

For operational use, the application should be integrated with
actual ERP/MRP data and should include additional constraints,
validation, security and appropriate optimization algorithms.

---

## Future Enhancements

Possible extensions:

- Upload Excel/CSV datasets
- Connect to ERPNext or another ERP system
- Integrate an LLM for managerial interpretation
- Multi-product and multi-machine scheduling
- Supplier risk analytics
- Monte Carlo demand simulation
- Production disruption simulation
- Machine downtime
- Workforce constraints
- Optimization of aggregate production plans
- Advanced forecasting methods
- Interactive scenario comparison
- Export management reports to PDF

---

## Disclaimer

This application is intended for educational purposes.

The results are illustrative and should not be treated as
operational production recommendations without validating
the underlying data, assumptions and constraints.
