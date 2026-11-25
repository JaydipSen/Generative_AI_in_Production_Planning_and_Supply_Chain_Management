
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from math import ceil, sqrt

# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PPC Decision Intelligence Lab",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 PPC Decision Intelligence Lab")
st.markdown(
    """
    **Generative AI in Production Planning and Supply Chain Management**

    An interactive management laboratory for demand forecasting,
    material planning, inventory, aggregate planning, capacity
    planning and production scheduling.
    """
)

# =========================================================
# SAMPLE DATA
# =========================================================

@st.cache_data
def generate_data():
    rng = np.random.default_rng(42)

    months = pd.date_range(
        end=pd.Timestamp.today().normalize(),
        periods=24,
        freq="MS"
    )

    base_demand = np.linspace(850, 1150, len(months))
    seasonal = 120 * np.sin(
        np.arange(len(months)) * 2 * np.pi / 12
    )
    noise = rng.normal(0, 65, len(months))

    demand = np.maximum(
        300,
        base_demand + seasonal + noise
    ).round().astype(int)

    sales = pd.DataFrame({
        "Month": months,
        "Demand": demand
    })

    products = pd.DataFrame({
        "Product": ["P-100", "P-200", "P-300", "P-400"],
        "Description": [
            "Metal Housing",
            "Industrial Shaft",
            "Mounting Bracket",
            "Precision Assembly"
        ],
        "Unit Selling Price": [2500, 1800, 950, 4200],
        "Unit Production Cost": [1500, 1050, 540, 2600],
        "Unit Labor Hours": [1.2, 0.8, 0.4, 2.0],
        "Machine Hours": [0.8, 1.1, 0.3, 1.5]
    })

    materials = pd.DataFrame({
        "Material": [
            "Steel Sheet",
            "Steel Rod",
            "Aluminium",
            "Fasteners",
            "Paint"
        ],
        "Unit": ["kg", "kg", "kg", "pcs", "litre"],
        "Unit Cost": [90, 110, 260, 4, 180],
        "Lead Time (weeks)": [3, 4, 2, 1, 2],
        "Available Stock": [2500, 1800, 900, 10000, 400],
        "Safety Stock": [500, 300, 150, 2000, 100],
        "Supplier": [
            "SteelCo", "RodWorks", "AluMet",
            "FastSupply", "PaintPro"
        ]
    })

    bom = pd.DataFrame({
        "Product": ["P-100", "P-100", "P-100",
                    "P-200", "P-200",
                    "P-300", "P-300",
                    "P-400", "P-400", "P-400"],
        "Material": [
            "Steel Sheet", "Fasteners", "Paint",
            "Steel Rod", "Fasteners",
            "Aluminium", "Fasteners",
            "Steel Sheet", "Aluminium", "Paint"
        ],
        "Quantity per Product": [
            2.5, 8, 0.15,
            1.8, 6,
            0.5, 4,
            3.5, 1.2, 0.25
        ]
    })

    work_centers = pd.DataFrame({
        "Work Center": [
            "Cutting", "Machining", "Welding",
            "Assembly", "Painting"
        ],
        "Available Hours/Month": [
            600, 800, 500, 700, 450
        ],
        "Efficiency": [
            0.85, 0.90, 0.80, 0.90, 0.85
        ]
    })

    jobs = pd.DataFrame({
        "Job": [f"JOB-{i:03}" for i in range(1, 13)],
        "Product": [
            "P-100", "P-200", "P-300", "P-400",
            "P-100", "P-300", "P-200", "P-400",
            "P-100", "P-200", "P-300", "P-400"
        ],
        "Quantity": [100, 150, 300, 60, 120, 250,
                    180, 70, 80, 200, 150, 50],
        "Processing Hours": [8, 12, 5, 15, 10, 4,
                            11, 16, 6, 10, 4, 14],
        "Due Day": [8, 12, 10, 15, 14, 11,
                    18, 20, 17, 22, 24, 25],
        "Release Day": [1, 1, 2, 3, 4, 5,
                        6, 7, 8, 9, 10, 11]
    })

    return sales, products, materials, bom, work_centers, jobs


sales, products, materials, bom, work_centers, jobs = generate_data()

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def money(x):
    return f"₹{x:,.0f}"


def forecast_series(series, method, window=3, alpha=0.3):
    """Forecast a one-step-ahead fitted series and future values."""

    s = pd.Series(series).astype(float).reset_index(drop=True)
    n = len(s)

    if method == "Moving Average":
        fitted = s.rolling(window).mean().shift(1)
        future = [s.tail(window).mean()] * 6

    elif method == "Weighted Moving Average":
        weights = np.arange(1, window + 1)
        weights = weights / weights.sum()

        fitted_values = [np.nan] * n

        for i in range(window, n):
            fitted_values[i] = np.dot(
                s.iloc[i-window:i].values,
                weights
            )

        fitted = pd.Series(fitted_values)

        future = []
        history = s.tolist()

        for _ in range(6):
            f = np.dot(history[-window:], weights)
            future.append(f)
            history.append(f)

    elif method == "Exponential Smoothing":
        fitted_values = [np.nan] * n
        level = s.iloc[0]

        for i in range(1, n):
            fitted_values[i] = level
            level = alpha * s.iloc[i] + (1-alpha) * level

        fitted = pd.Series(fitted_values)

        future = [level] * 6

    elif method == "Trend":
        x = np.arange(n)
        coef = np.polyfit(x, s.values, 1)
        fitted = pd.Series(np.polyval(coef, x))
        future_x = np.arange(n, n+6)
        future = np.polyval(coef, future_x).tolist()

    else:
        fitted = pd.Series([np.nan] * n)
        future = [s.iloc[-1]] * 6

    return fitted, future


def calc_metrics(actual, fitted):
    valid = pd.DataFrame({
        "Actual": actual,
        "Forecast": fitted
    }).dropna()

    if len(valid) == 0:
        return np.nan, np.nan, np.nan

    errors = valid["Actual"] - valid["Forecast"]

    mae = errors.abs().mean()
    rmse = np.sqrt((errors ** 2).mean())

    denominator = valid["Actual"].replace(0, np.nan)
    mape = (errors.abs() / denominator).mean() * 100

    return mae, rmse, mape


def calculate_eoq(demand, ordering_cost, holding_cost):
    if holding_cost <= 0:
        return 0
    return sqrt((2 * demand * ordering_cost) / holding_cost)


def safety_stock(z, demand_std, lead_time):
    return z * demand_std * sqrt(max(lead_time, 0))


def reorder_point(avg_demand, lead_time, safety):
    return avg_demand * lead_time + safety


def abc_analysis(df):
    d = df.copy()
    d["Annual Value"] = (
        d["Annual Demand"] * d["Unit Cost"]
    )
    d = d.sort_values("Annual Value", ascending=False)
    d["Cumulative %"] = (
        d["Annual Value"].cumsum()
        / d["Annual Value"].sum() * 100
    )

    d["ABC"] = np.where(
        d["Cumulative %"] <= 80, "A",
        np.where(d["Cumulative %"] <= 95, "B", "C")
    )

    return d


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Navigation")

module = st.sidebar.radio(
    "Select PPC module",
    [
        "Overview",
        "Demand Forecasting",
        "MRP",
        "Inventory Management",
        "Aggregate Planning",
        "Capacity Planning",
        "Job-shop Scheduling",
        "AI Decision Support"
    ]
)

st.sidebar.divider()
st.sidebar.caption(
    "Illustrative teaching data. Replace with ERP, PPC "
    "or actual manufacturing data when available."
)

# =========================================================
# OVERVIEW
# =========================================================

if module == "Overview":

    st.header("PPC Overview Dashboard")

    total_demand = sales["Demand"].sum()
    avg_monthly_demand = sales["Demand"].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Historical Demand", f"{total_demand:,}")
    c2.metric("Average Monthly Demand", f"{avg_monthly_demand:,.0f}")
    c3.metric("Products", len(products))
    c4.metric("Materials", len(materials))

    fig = px.line(
        sales,
        x="Month",
        y="Demand",
        markers=True,
        title="Historical Monthly Demand"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Manufacturing Product Portfolio")
    st.dataframe(products, use_container_width=True)

    st.subheader("Material Master")
    st.dataframe(materials, use_container_width=True)

    st.subheader("Bill of Materials")
    st.dataframe(bom, use_container_width=True)

    st.subheader("Work-centre Master")
    st.dataframe(work_centers, use_container_width=True)

# =========================================================
# DEMAND FORECASTING
# =========================================================

elif module == "Demand Forecasting":

    st.header("Demand Forecasting Laboratory")

    st.write(
        "Explore forecasting techniques and compare "
        "forecast accuracy."
    )

    method = st.selectbox(
        "Forecasting method",
        [
            "Moving Average",
            "Weighted Moving Average",
            "Exponential Smoothing",
            "Trend"
        ]
    )

    window = st.slider(
        "Moving average window",
        2, 12, 3
    )

    alpha = st.slider(
        "Exponential smoothing alpha",
        0.05, 0.95, 0.30
    )

    fitted, future = forecast_series(
        sales["Demand"],
        method,
        window,
        alpha
    )

    mae, rmse, mape = calc_metrics(
        sales["Demand"],
        fitted
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "MAE",
        f"{mae:,.1f}" if not np.isnan(mae) else "N/A"
    )

    c2.metric(
        "RMSE",
        f"{rmse:,.1f}" if not np.isnan(rmse) else "N/A"
    )

    c3.metric(
        "MAPE",
        f"{mape:.2f}%" if not np.isnan(mape) else "N/A"
    )

    historical = pd.DataFrame({
        "Month": sales["Month"],
        "Actual Demand": sales["Demand"],
        "Fitted Forecast": fitted
    })

    future_dates = pd.date_range(
        sales["Month"].iloc[-1] + pd.offsets.MonthBegin(1),
        periods=6,
        freq="MS"
    )

    forecast_df = pd.DataFrame({
        "Month": future_dates,
        "Forecast": future
    })

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=historical["Month"],
        y=historical["Actual Demand"],
        name="Actual Demand",
        mode="lines+markers"
    ))

    fig.add_trace(go.Scatter(
        x=historical["Month"],
        y=historical["Fitted Forecast"],
        name="Fitted Forecast",
        mode="lines",
        line=dict(dash="dash")
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["Month"],
        y=forecast_df["Forecast"],
        name="Future Forecast",
        mode="lines+markers",
        line=dict(dash="dot")
    ))

    fig.update_layout(
        title="Actual Demand, Fitted Forecast and Future Forecast",
        xaxis_title="Month",
        yaxis_title="Units"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Data")

    st.dataframe(
        forecast_df.style.format({
            "Forecast": "{:,.0f}"
        }),
        use_container_width=True
    )

    # Forecast error analysis
    historical["Error"] = (
        historical["Actual Demand"]
        - historical["Fitted Forecast"]
    )

    st.subheader("Forecast Error Analysis")

    error_fig = px.bar(
        historical.dropna(),
        x="Month",
        y="Error",
        title="Forecast Error by Month"
    )

    error_fig.add_hline(y=0, line_dash="dash")

    st.plotly_chart(error_fig, use_container_width=True)

    st.info(
        "Teaching insight: forecasting accuracy should be "
        "evaluated alongside business consequences. A forecast "
        "that underestimates demand may cause stockouts, while "
        "overestimation may create excess inventory."
    )

# =========================================================
# MRP
# =========================================================

elif module == "MRP":

    st.header("Material Requirements Planning (MRP)")

    st.markdown(
        "Explode the Bill of Materials and estimate material "
        "requirements for a planned production quantity."
    )

    product_choice = st.selectbox(
        "Select product",
        products["Product"].tolist()
    )

    production_qty = st.number_input(
        "Planned production quantity",
        min_value=1,
        value=500,
        step=50
    )

    selected_bom = bom[
        bom["Product"] == product_choice
    ].copy()

    selected_bom["Gross Requirement"] = (
        selected_bom["Quantity per Product"]
        * production_qty
    )

    selected_bom = selected_bom.merge(
        materials,
        on="Material",
        how="left"
    )

    selected_bom["Net Requirement"] = np.maximum(
        0,
        selected_bom["Gross Requirement"]
        - selected_bom["Available Stock"]
    )

    selected_bom["Projected Stock"] = (
        selected_bom["Available Stock"]
        - selected_bom["Gross Requirement"]
    )

    selected_bom["Procurement Cost"] = (
        selected_bom["Net Requirement"]
        * selected_bom["Unit Cost"]
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Production Quantity",
        f"{production_qty:,}"
    )

    c2.metric(
        "Materials Required",
        f"{len(selected_bom)}"
    )

    c3.metric(
        "Estimated Procurement Cost",
        money(selected_bom["Procurement Cost"].sum())
    )

    st.subheader("Material Requirements")

    st.dataframe(
        selected_bom[[
            "Material",
            "Quantity per Product",
            "Gross Requirement",
            "Available Stock",
            "Net Requirement",
            "Lead Time (weeks)",
            "Unit Cost",
            "Procurement Cost"
        ]],
        use_container_width=True
    )

    st.subheader("Gross Requirement vs Available Stock")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=selected_bom["Material"],
        y=selected_bom["Gross Requirement"],
        name="Gross Requirement"
    ))

    fig.add_trace(go.Bar(
        x=selected_bom["Material"],
        y=selected_bom["Available Stock"],
        name="Available Stock"
    ))

    fig.update_layout(
        barmode="group",
        title="Material Requirements Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)

    shortages = selected_bom[
        selected_bom["Net Requirement"] > 0
    ]

    st.subheader("Procurement Recommendations")

    if shortages.empty:
        st.success(
            "Available inventory is sufficient for the "
            "selected production quantity."
        )
    else:
        st.warning(
            f"{len(shortages)} materials require procurement."
        )

        st.dataframe(
            shortages[[
                "Material",
                "Net Requirement",
                "Lead Time (weeks)",
                "Supplier",
                "Procurement Cost"
            ]],
            use_container_width=True
        )

        st.markdown(
            """
            **Recommended managerial actions**

            - Initiate purchase orders for net shortages.
            - Prioritize long-lead-time materials.
            - Verify supplier delivery commitments.
            - Check whether production should be rescheduled.
            - Consider alternative suppliers for critical materials.
            """
        )

# =========================================================
# INVENTORY MANAGEMENT
# =========================================================

elif module == "Inventory Management":

    st.header("Inventory Management and Replenishment")

    st.write(
        "Calculate EOQ, safety stock, reorder point, "
        "inventory days and ABC classification."
    )

    ordering_cost = st.number_input(
        "Ordering cost per order (₹)",
        min_value=100,
        value=1500,
        step=500
    )

    annual_holding_rate = st.slider(
        "Annual inventory carrying cost (%)",
        5, 40, 20
    ) / 100

    service_level = st.selectbox(
        "Target service level",
        [90, 95, 97, 98, 99]
    )

    z_values = {
        90: 1.282,
        95: 1.645,
        97: 1.881,
        98: 2.054,
        99: 2.326
    }

    z = z_values[service_level]

    inventory_df = materials.copy()

    inventory_df["Annual Demand"] = [
        15000, 10000, 7000, 50000, 2500
    ]

    inventory_df["Daily Demand"] = (
        inventory_df["Annual Demand"] / 365
    )

    inventory_df["Holding Cost"] = (
        inventory_df["Unit Cost"] * annual_holding_rate
    )

    inventory_df["EOQ"] = inventory_df.apply(
        lambda r: calculate_eoq(
            r["Annual Demand"],
            ordering_cost,
            r["Holding Cost"]
        ),
        axis=1
    )

    # Illustrative demand variability
    inventory_df["Daily Demand Std"] = (
        inventory_df["Daily Demand"] * 0.25
    )

    inventory_df["Safety Stock"] = inventory_df.apply(
        lambda r: safety_stock(
            z,
            r["Daily Demand Std"],
            r["Lead Time (weeks)"] * 7
        ),
        axis=1
    )

    inventory_df["Reorder Point"] = inventory_df.apply(
        lambda r: reorder_point(
            r["Daily Demand"],
            r["Lead Time (weeks)"] * 7,
            r["Safety Stock"]
        ),
        axis=1
    )

    inventory_df["Days of Inventory"] = (
        inventory_df["Available Stock"]
        / inventory_df["Daily Demand"]
    )

    inventory_df["Annual Value"] = (
        inventory_df["Annual Demand"]
        * inventory_df["Unit Cost"]
    )

    abc_df = abc_analysis(inventory_df)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Inventory Value",
        money(
            (
                inventory_df["Available Stock"]
                * inventory_df["Unit Cost"]
            ).sum()
        )
    )

    c2.metric(
        "Target Service Level",
        f"{service_level}%"
    )

    c3.metric(
        "Items Below Reorder Point",
        int(
            (
                inventory_df["Available Stock"]
                < inventory_df["Reorder Point"]
            ).sum()
        )
    )

    st.subheader("Inventory Control Calculations")

    st.dataframe(
        inventory_df[[
            "Material",
            "Available Stock",
            "Annual Demand",
            "EOQ",
            "Safety Stock",
            "Reorder Point",
            "Days of Inventory"
        ]].style.format({
            "EOQ": "{:,.0f}",
            "Safety Stock": "{:,.0f}",
            "Reorder Point": "{:,.0f}",
            "Days of Inventory": "{:.1f}"
        }),
        use_container_width=True
    )

    # Inventory and reorder point visualization
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=inventory_df["Material"],
        y=inventory_df["Available Stock"],
        name="Available Stock"
    ))

    fig.add_trace(go.Scatter(
        x=inventory_df["Material"],
        y=inventory_df["Reorder Point"],
        name="Reorder Point",
        mode="markers",
        marker=dict(size=12)
    ))

    fig.update_layout(
        title="Inventory Position versus Reorder Point"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("ABC Inventory Classification")

    abc_fig = px.bar(
        abc_df,
        x="Material",
        y="Annual Value",
        color="ABC",
        title="Annual Consumption Value by Material"
    )

    st.plotly_chart(abc_fig, use_container_width=True)

    st.dataframe(
        abc_df[[
            "Material",
            "Annual Demand",
            "Unit Cost",
            "Annual Value",
            "Cumulative %",
            "ABC"
        ]],
        use_container_width=True
    )

    st.markdown("""
    **Managerial interpretation**

    - A-items deserve tighter monitoring and purchasing controls.
    - B-items require moderate control.
    - C-items may be managed with simpler replenishment rules.
    - Reorder points should be reviewed whenever lead time or demand changes.
    """)

# =========================================================
# AGGREGATE PLANNING
# =========================================================

elif module == "Aggregate Planning":

    st.header("Aggregate Production Planning")

    planning_horizon = st.slider(
        "Planning horizon (months)",
        3, 12, 6
    )

    monthly_forecast = sales["Demand"].tail(
        planning_horizon
    ).values

    beginning_inventory = st.number_input(
        "Beginning inventory (units)",
        min_value=0,
        value=500
    )

    workforce_initial = st.number_input(
        "Initial workforce",
        min_value=1,
        value=25
    )

    productivity = st.number_input(
        "Units per worker per month",
        min_value=1,
        value=40
    )

    regular_cost = st.number_input(
        "Monthly cost per worker (₹)",
        min_value=1000,
        value=30000
    )

    hiring_cost = st.number_input(
        "Hiring cost per worker (₹)",
        min_value=0,
        value=5000
    )

    firing_cost = st.number_input(
        "Layoff cost per worker (₹)",
        min_value=0,
        value=8000
    )

    inventory_cost = st.number_input(
        "Inventory holding cost per unit per month (₹)",
        min_value=0,
        value=20
    )

    shortage_cost = st.number_input(
        "Backorder cost per unit (₹)",
        min_value=0,
        value=100
    )

    plan_df = pd.DataFrame({
        "Month": range(1, planning_horizon + 1),
        "Demand": monthly_forecast
    })

    # LEVEL PLAN
    level_production = int(
        np.ceil(np.mean(monthly_forecast))
    )

    level_inventory = beginning_inventory
    level_results = []

    for d in monthly_forecast:
        level_inventory += level_production - d

        level_results.append({
            "Production": level_production,
            "Inventory": level_inventory,
            "Shortage": max(0, -level_inventory)
        })

    level_df = pd.DataFrame(level_results)

    # CHASE PLAN
    chase_results = []
    chase_inventory = beginning_inventory
    chase_workforce = workforce_initial

    for d in monthly_forecast:
        required_production = max(
            0, d - chase_inventory
        )

        required_workers = ceil(
            required_production / productivity
        )

        production = required_workers * productivity

        chase_inventory += production - d

        chase_results.append({
            "Production": production,
            "Workforce": required_workers,
            "Inventory": chase_inventory,
            "Shortage": max(0, -chase_inventory)
        })

        chase_workforce = required_workers

    chase_df = pd.DataFrame(chase_results)

    # HYBRID PLAN
    hybrid_production = int(
        np.ceil(np.mean(monthly_forecast) * 0.9)
    )

    hybrid_results = []
    hybrid_inventory = beginning_inventory
    hybrid_workforce = workforce_initial

    for d in monthly_forecast:
        hybrid_inventory += hybrid_production - d

        required_workers = ceil(
            hybrid_production / productivity
        )

        hybrid_results.append({
            "Production": hybrid_production,
            "Workforce": required_workers,
            "Inventory": hybrid_inventory,
            "Shortage": max(0, -hybrid_inventory)
        })

    hybrid_df = pd.DataFrame(hybrid_results)

    def plan_cost(production, workforce, inventory, shortage):
        regular = np.sum(workforce) * regular_cost

        inv_cost = (
            np.maximum(inventory, 0).sum()
            * inventory_cost
        )

        shortage_total = (
            np.maximum(shortage, 0).sum()
            * shortage_cost
        )

        changes = np.diff(
            np.concatenate(([workforce_initial], workforce))
        )

        hiring = np.maximum(changes, 0).sum() * hiring_cost
        firing = np.maximum(-changes, 0).sum() * firing_cost

        return regular + inv_cost + shortage_total + hiring + firing

    level_workforce = np.repeat(
        ceil(level_production / productivity),
        planning_horizon
    )

    level_cost = plan_cost(
        level_df["Production"].values,
        level_workforce,
        level_df["Inventory"].values,
        level_df["Shortage"].values
    )

    chase_cost = plan_cost(
        chase_df["Production"].values,
        chase_df["Workforce"].values,
        chase_df["Inventory"].values,
        chase_df["Shortage"].values
    )

    hybrid_cost = plan_cost(
        hybrid_df["Production"].values,
        hybrid_df["Workforce"].values,
        hybrid_df["Inventory"].values,
        hybrid_df["Shortage"].values
    )

    costs = pd.DataFrame({
        "Strategy": ["Level", "Chase", "Hybrid"],
        "Estimated Total Cost": [
            level_cost, chase_cost, hybrid_cost
        ]
    })

    st.subheader("Plan Cost Comparison")

    cost_fig = px.bar(
        costs,
        x="Strategy",
        y="Estimated Total Cost",
        text_auto=".2s",
        title="Estimated Cost of Aggregate Plans"
    )

    st.plotly_chart(cost_fig, use_container_width=True)

    st.dataframe(
        costs.style.format({
            "Estimated Total Cost": "₹{:,.0f}"
        }),
        use_container_width=True
    )

    # Production and demand
    plan_comparison = pd.DataFrame({
        "Month": range(1, planning_horizon + 1),
        "Demand": monthly_forecast,
        "Level Production": level_df["Production"],
        "Chase Production": chase_df["Production"],
        "Hybrid Production": hybrid_df["Production"]
    })

    fig = px.line(
        plan_comparison,
        x="Month",
        y=[
            "Demand",
            "Level Production",
            "Chase Production",
            "Hybrid Production"
        ],
        markers=True,
        title="Demand versus Production Plans"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Inventory profiles
    inv_comparison = pd.DataFrame({
        "Month": range(1, planning_horizon + 1),
        "Level": level_df["Inventory"],
        "Chase": chase_df["Inventory"],
        "Hybrid": hybrid_df["Inventory"]
    })

    inv_fig = px.line(
        inv_comparison,
        x="Month",
        y=["Level", "Chase", "Hybrid"],
        markers=True,
        title="Inventory Evolution under Different Plans"
    )

    st.plotly_chart(inv_fig, use_container_width=True)

    st.subheader("Plan Details")

    st.write("Level Plan")
    st.dataframe(level_df, use_container_width=True)

    st.write("Chase Plan")
    st.dataframe(chase_df, use_container_width=True)

    st.write("Hybrid Plan")
    st.dataframe(hybrid_df, use_container_width=True)

    st.info(
        "These plans are illustrative and do not include "
        "all operational constraints. Cost comparisons are "
        "sensitive to workforce, production, inventory and "
        "shortage-cost assumptions."
    )

# =========================================================
# CAPACITY PLANNING
# =========================================================

elif module == "Capacity Planning":

    st.header("Capacity Requirements Planning")

    st.write(
        "Estimate workload and identify work centres "
        "with insufficient capacity."
    )

    demand_multiplier = st.slider(
        "Demand scenario multiplier",
        0.50, 2.00, 1.00, 0.05
    )

    capacity_df = work_centers.copy()

    capacity_df["Demand Multiplier"] = demand_multiplier

    # Illustrative workload for each work centre
    base_workload = np.array([
        560, 720, 480, 650, 400
    ])

    capacity_df["Workload Hours"] = (
        base_workload * demand_multiplier
    )

    capacity_df["Effective Capacity"] = (
        capacity_df["Available Hours/Month"]
        * capacity_df["Efficiency"]
    )

    capacity_df["Utilization (%)"] = (
        capacity_df["Workload Hours"]
        / capacity_df["Effective Capacity"]
        * 100
    )

    capacity_df["Overload Hours"] = np.maximum(
        0,
        capacity_df["Workload Hours"]
        - capacity_df["Effective Capacity"]
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Work Centres",
        len(capacity_df)
    )

    c2.metric(
        "Bottleneck",
        capacity_df.loc[
            capacity_df["Utilization (%)"].idxmax(),
            "Work Center"
        ]
    )

    c3.metric(
        "Maximum Utilization",
        f"{capacity_df['Utilization (%)'].max():.1f}%"
    )

    st.subheader("Work-centre Capacity Analysis")

    st.dataframe(
        capacity_df.style.format({
            "Workload Hours": "{:,.1f}",
            "Effective Capacity": "{:,.1f}",
            "Utilization (%)": "{:.1f}%",
            "Overload Hours": "{:,.1f}"
        }),
        use_container_width=True
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=capacity_df["Work Center"],
        y=capacity_df["Effective Capacity"],
        name="Effective Capacity"
    ))

    fig.add_trace(go.Bar(
        x=capacity_df["Work Center"],
        y=capacity_df["Workload Hours"],
        name="Workload"
    ))

    fig.update_layout(
        barmode="group",
        title="Workload versus Effective Capacity"
    )

    st.plotly_chart(fig, use_container_width=True)

    utilization_fig = px.bar(
        capacity_df,
        x="Work Center",
        y="Utilization (%)",
        color="Utilization (%)",
        title="Work-centre Utilization"
    )

    utilization_fig.add_hline(
        y=100,
        line_dash="dash",
        annotation_text="100% capacity"
    )

    st.plotly_chart(
        utilization_fig,
        use_container_width=True
    )

    overloaded = capacity_df[
        capacity_df["Utilization (%)"] > 100
    ]

    if not overloaded.empty:
        st.warning(
            "Capacity overload detected in one or more "
            "work centres."
        )

        st.dataframe(
            overloaded[[
                "Work Center",
                "Workload Hours",
                "Effective Capacity",
                "Overload Hours"
            ]],
            use_container_width=True
        )

        st.markdown("""
        **Possible managerial actions**

        - Authorize overtime.
        - Shift work to alternative work centres.
        - Outsource selected operations.
        - Reschedule lower-priority orders.
        - Add shifts or equipment.
        - Reduce setup time.
        """)

# =========================================================
# JOB-SHOP SCHEDULING
# =========================================================

elif module == "Job-shop Scheduling":

    st.header("Job-shop Scheduling Laboratory")

    st.write(
        "Compare FCFS, SPT and EDD sequencing rules "
        "for jobs processed on a single constrained machine."
    )

    sequence_rule = st.selectbox(
        "Scheduling rule",
        [
            "FCFS - First Come First Served",
            "SPT - Shortest Processing Time",
            "EDD - Earliest Due Date"
        ]
    )

    schedule_jobs = jobs.copy()

    if sequence_rule.startswith("FCFS"):
        schedule_jobs = schedule_jobs.sort_values(
            "Release Day"
        )

    elif sequence_rule.startswith("SPT"):
        schedule_jobs = schedule_jobs.sort_values(
            "Processing Hours"
        )

    elif sequence_rule.startswith("EDD"):
        schedule_jobs = schedule_jobs.sort_values(
            "Due Day"
        )

    current_time = 0
    schedule_results = []

    for _, job in schedule_jobs.iterrows():

        start = max(
            current_time,
            job["Release Day"]
        )

        finish = start + job["Processing Hours"]

        tardiness = max(
            0,
            finish - job["Due Day"]
        )

        schedule_results.append({
            "Job": job["Job"],
            "Product": job["Product"],
            "Quantity": job["Quantity"],
            "Processing Hours": job["Processing Hours"],
            "Due Day": job["Due Day"],
            "Start": start,
            "Finish": finish,
            "Tardiness": tardiness
        })

        current_time = finish

    schedule_df = pd.DataFrame(schedule_results)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Processing Time",
        f"{schedule_df['Processing Hours'].sum():.1f} hours"
    )

    c2.metric(
        "Average Tardiness",
        f"{schedule_df['Tardiness'].mean():.2f} days"
    )

    c3.metric(
        "Total Tardiness",
        f"{schedule_df['Tardiness'].sum():.1f} days"
    )

    st.subheader("Schedule")

    st.dataframe(
        schedule_df,
        use_container_width=True
    )

    st.subheader("Production Schedule Gantt Chart")

    gantt = px.timeline(
        schedule_df,
        x_start="Start",
        x_end="Finish",
        y="Job",
        color="Product",
        hover_data=[
            "Quantity",
            "Due Day",
            "Tardiness"
        ],
        title=f"Job Schedule using {sequence_rule}"
    )

    gantt.update_yaxes(
        autorange="reversed"
    )

    st.plotly_chart(
        gantt,
        use_container_width=True
    )

    st.subheader("Due Date versus Completion")

    due_fig = go.Figure()

    due_fig.add_trace(go.Bar(
        x=schedule_df["Job"],
        y=schedule_df["Due Day"],
        name="Due Day"
    ))

    due_fig.add_trace(go.Bar(
        x=schedule_df["Job"],
        y=schedule_df["Finish"],
        name="Completion Day"
    ))

    due_fig.update_layout(
        barmode="group",
        title="Job Due Dates and Scheduled Completion"
    )

    st.plotly_chart(
        due_fig,
        use_container_width=True
    )

    st.info(
        "Scheduling rules optimize different objectives. "
        "SPT often reduces average completion time, while "
        "EDD prioritizes due dates. FCFS follows release order. "
        "The preferred rule depends on the production objective."
    )

# =========================================================
# AI DECISION SUPPORT
# =========================================================

elif module == "AI Decision Support":

    st.header("AI-assisted PPC Decision Support")

    st.markdown(
        """
        This module converts analytical outputs into
        managerial explanations and recommendations.
        The interpretation is rule-based and runs locally;
        it is not a live LLM connection.
        """
    )

    st.subheader("Select a decision problem")

    decision = st.selectbox(
        "Decision area",
        [
            "Demand Forecasting",
            "Material Shortage",
            "Inventory Risk",
            "Capacity Bottleneck",
            "Production Scheduling"
        ]
    )

    if decision == "Demand Forecasting":

        method = st.selectbox(
            "Forecast method",
            [
                "Moving Average",
                "Weighted Moving Average",
                "Exponential Smoothing",
                "Trend"
            ]
        )

        fitted, future = forecast_series(
            sales["Demand"],
            method
        )

        mae, rmse, mape = calc_metrics(
            sales["Demand"],
            fitted
        )

        st.markdown("### Managerial interpretation")

        st.write(
            f"The selected forecasting technique is {method}. "
            f"The estimated MAE is {mae:.1f} units and "
            f"RMSE is {rmse:.1f} units."
        )

        if not np.isnan(mape):
            st.write(
                f"The estimated MAPE is {mape:.1f}%. "
            )

            if mape < 10:
                st.success(
                    "Forecast errors are relatively low "
                    "for this illustrative dataset."
                )
            elif mape < 20:
                st.info(
                    "Forecast accuracy is moderate. "
                    "Review demand variability and seasonality."
                )
            else:
                st.warning(
                    "Forecast errors are substantial. "
                    "Review the forecasting model, demand "
                    "drivers and possible outliers."
                )

        st.markdown("""
        **Recommended management actions**

        - Compare forecasting techniques.
        - Review demand variability.
        - Identify periods of unusual demand.
        - Adjust production and inventory plans.
        - Avoid treating historical forecasts as certainty.
        """)

    elif decision == "Material Shortage":

        product = st.selectbox(
            "Product",
            products["Product"].tolist()
        )

        quantity = st.number_input(
            "Production quantity",
            min_value=1,
            value=500
        )

        mrp = bom[bom["Product"] == product].copy()

        mrp["Requirement"] = (
            mrp["Quantity per Product"] * quantity
        )

        mrp = mrp.merge(
            materials,
            on="Material"
        )

        mrp["Shortage"] = np.maximum(
            0,
            mrp["Requirement"]
            - mrp["Available Stock"]
        )

        shortages = mrp[
            mrp["Shortage"] > 0
        ]

        if shortages.empty:
            st.success(
                "No material shortage identified "
                "for the selected production quantity."
            )
        else:
            st.warning(
                f"{len(shortages)} materials require replenishment."
            )

            st.dataframe(
                shortages[[
                    "Material",
                    "Requirement",
                    "Available Stock",
                    "Shortage",
                    "Lead Time (weeks)",
                    "Supplier"
                ]],
                use_container_width=True
            )

            st.markdown("""
            **Recommendations**

            1. Place purchase orders for shortage quantities.
            2. Review long-lead-time materials first.
            3. Confirm delivery dates with suppliers.
            4. Assess production rescheduling.
            5. Investigate alternative sources.
            """)

    elif decision == "Inventory Risk":

        risk_df = materials.copy()

        risk_df["Annual Demand"] = [
            15000, 10000, 7000, 50000, 2500
        ]

        risk_df["Daily Demand"] = (
            risk_df["Annual Demand"] / 365
        )

        risk_df["Estimated Reorder Point"] = (
            risk_df["Daily Demand"]
            * risk_df["Lead Time (weeks)"] * 7
            + risk_df["Safety Stock"]
        )

        risk_df["Risk"] = np.where(
            risk_df["Available Stock"]
            < risk_df["Estimated Reorder Point"],
            "Replenishment Risk",
            "Adequate"
        )

        st.dataframe(
            risk_df[[
                "Material",
                "Available Stock",
                "Estimated Reorder Point",
                "Risk"
            ]],
            use_container_width=True
        )

        risks = risk_df[
            risk_df["Risk"] == "Replenishment Risk"
        ]

        if len(risks) > 0:
            st.warning(
                f"{len(risks)} materials may require "
                "replenishment attention."
            )
        else:
            st.success(
                "No immediate reorder-point risk "
                "identified under the current assumptions."
            )

    elif decision == "Capacity Bottleneck":

        cap = work_centers.copy()

        cap["Workload"] = [
            560, 720, 480, 650, 400
        ]

        cap["Effective Capacity"] = (
            cap["Available Hours/Month"]
            * cap["Efficiency"]
        )

        cap["Utilization"] = (
            cap["Workload"]
            / cap["Effective Capacity"]
        )

        bottleneck = cap.loc[
            cap["Utilization"].idxmax()
        ]

        st.warning(
            f"Potential bottleneck: {bottleneck['Work Center']} "
            f"with utilization of "
            f"{bottleneck['Utilization']*100:.1f}%."
        )

        st.markdown("""
        **Possible recommendations**

        - Investigate overtime.
        - Improve process efficiency.
        - Shift jobs to alternative work centres.
        - Outsource selected operations.
        - Reconsider production priorities.
        """)

    elif decision == "Production Scheduling":

        st.write(
            "Use the Job-shop Scheduling module to compare "
            "FCFS, SPT and EDD and evaluate completion time "
            "and tardiness."
        )

        st.markdown("""
        **Management should consider**

        - Customer due dates.
        - Production priorities.
        - Setup requirements.
        - Bottleneck resources.
        - Machine availability.
        - Order criticality.
        """)

    st.divider()

    st.subheader("Managerial Decision Framework")

    st.markdown("""
    ### Before implementing an AI recommendation, managers should:

    1. **Validate the data** — Is the input reliable and current?
    2. **Understand the assumptions** — Which constraints are included?
    3. **Evaluate trade-offs** — Cost, service, capacity and risk.
    4. **Compare alternatives** — Is there a better plan?
    5. **Use managerial judgment** — Consider strategic and operational factors.
    6. **Monitor outcomes** — Compare recommendations with actual results.
    """)

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "PPC Decision Intelligence Lab | Educational application | "
    "Illustrative data | Not intended as an ERP replacement."
)
