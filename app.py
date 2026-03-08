"""
PizzaIQ — Sales Intelligence Platform
Flask Backend: Data Processing + ML Forecasting API
Run: python app.py  →  http://localhost:5000
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import json
import io
from datetime import datetime, timedelta

app = Flask(__name__)
# ─────────────────────────────────────────────
# CORS (manual, no extra package needed)
# ─────────────────────────────────────────────
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# FIXED: Removed the (path) variable and renamed for clarity
@app.route('/', methods=['GET', 'POST', 'OPTIONS'])
def home_route():
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200
    return jsonify({"message": "PizzaIQ API is running"}), 200


# ═══════════════════════════════════════════════
#  CSV ANALYTICS ENGINE
# ═══════════════════════════════════════════════

def process_csv(df: pd.DataFrame) -> dict:
    """
    Transform a raw pizza-sales CSV into the full analytics payload.
    Expected columns: order_id, order_date, order_time, total_price,
                      pizza_size, pizza_category, pizza_name, quantity
    """
    # ── normalise column names ──────────────────
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    required = {"order_id", "order_date", "total_price",
                "pizza_size", "pizza_category", "pizza_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    # ── parse date / time ───────────────────────
    # This version handles both '-' and '/' separators and infers the correct order
    df["order_date"] = pd.to_datetime(df["order_date"], dayfirst=True, errors="coerce")
    if "order_time" in df.columns:
        df["hour"] = pd.to_datetime(
            df["order_date"].astype(str) + " " + df["order_time"].astype(str),
            errors="coerce"
        ).dt.hour
    else:
        df["hour"] = 12  # fallback

    df["dow"]       = df["order_date"].dt.dayofweek          # 0=Mon
    df["week"]      = df["order_date"].dt.strftime("%G-W%V")
    df["month"]     = df["order_date"].dt.strftime("%Y-%m")
    df["date_str"]  = df["order_date"].dt.strftime("%Y-%m-%d")

    qty_col = "quantity" if "quantity" in df.columns else None

    # ── summary ────────────────────────────────
    total_revenue  = float(df["total_price"].sum())
    total_orders   = int(df["order_id"].nunique())
    total_pizzas   = int(df[qty_col].sum()) if qty_col else int(len(df))
    date_range_str = (
        f"{df['order_date'].min().strftime('%Y-%m-%d')} to "
        f"{df['order_date'].max().strftime('%Y-%m-%d')}"
    )
    num_pizza_types = int(df["pizza_name"].nunique())
    num_days        = int(df["date_str"].nunique())
    avg_daily_rev   = round(total_revenue / max(num_days, 1), 2)
    avg_order_value = round(total_revenue / max(total_orders, 1), 2)

    summary = {
        "total_revenue":    round(total_revenue, 2),
        "total_orders":     total_orders,
        "total_pizzas":     total_pizzas,
        "avg_daily_revenue": avg_daily_rev,
        "avg_order_value":  avg_order_value,
        "date_range":       date_range_str,
        "num_pizza_types":  num_pizza_types,
    }

    # ── daily ───────────────────────────────────
    daily_g = (
        df.groupby("date_str")
          .agg(revenue=("total_price", "sum"), orders=("order_id", "nunique"))
          .sort_index()
          .reset_index()
    )
    daily = {
        "dates":   daily_g["date_str"].tolist(),
        "revenue": [round(v, 2) for v in daily_g["revenue"].tolist()],
        "orders":  daily_g["orders"].tolist(),
    }

    # 7-day moving average
    rev_series = pd.Series(daily_g["revenue"].values)
    ma7 = rev_series.rolling(7, min_periods=1).mean()
    daily["ma7"] = [round(v, 2) for v in ma7.tolist()]

    # ── weekly ──────────────────────────────────
    weekly_g = (
        df.groupby("week")
          .agg(revenue=("total_price", "sum"), orders=("order_id", "nunique"))
          .sort_index()
          .reset_index()
    )
    weekly = {
        "weeks":   weekly_g["week"].tolist(),
        "revenue": [round(v, 2) for v in weekly_g["revenue"].tolist()],
        "orders":  weekly_g["orders"].tolist(),
    }

    # ── monthly ─────────────────────────────────
    monthly_g = (
        df.groupby("month")
          .agg(revenue=("total_price", "sum"), orders=("order_id", "nunique"))
          .sort_index()
          .reset_index()
    )
    monthly = {
        "months":  monthly_g["month"].tolist(),
        "revenue": [round(v, 2) for v in monthly_g["revenue"].tolist()],
        "orders":  monthly_g["orders"].tolist(),
    }

    # ── category ────────────────────────────────
    cat_g = (
        df.groupby("pizza_category")
          .agg(revenue=("total_price", "sum"),
               qty=(qty_col if qty_col else "order_id", "sum" if qty_col else "count"))
          .reset_index()
          .sort_values("revenue", ascending=False)
    )
    category = {
        "names":   cat_g["pizza_category"].tolist(),
        "revenue": [round(v, 2) for v in cat_g["revenue"].tolist()],
        "qty":     cat_g["qty"].tolist(),
    }

    # ── size ────────────────────────────────────
    size_order = {"S": 0, "M": 1, "L": 2, "XL": 3, "XXL": 4}
    size_g = (
        df.groupby("pizza_size")
          .agg(revenue=("total_price", "sum"),
               qty=(qty_col if qty_col else "order_id", "sum" if qty_col else "count"))
          .reset_index()
    )
    size_g["_ord"] = size_g["pizza_size"].map(lambda x: size_order.get(x, 99))
    size_g = size_g.sort_values("revenue", ascending=False)
    size = {
        "names":   size_g["pizza_size"].tolist(),
        "revenue": [round(v, 2) for v in size_g["revenue"].tolist()],
        "qty":     size_g["qty"].tolist(),
    }

    # ── top pizzas by revenue ───────────────────
    rev_by_pizza = (
        df.groupby("pizza_name")["total_price"]
          .sum()
          .sort_values(ascending=False)
          .head(10)
    )
    top_pizzas_rev = {
        "names":  rev_by_pizza.index.tolist(),
        "values": [round(v, 2) for v in rev_by_pizza.values.tolist()],
    }

    # ── top pizzas by quantity ──────────────────
    if qty_col:
        qty_by_pizza = (
            df.groupby("pizza_name")[qty_col]
              .sum()
              .sort_values(ascending=False)
              .head(10)
        )
    else:
        qty_by_pizza = (
            df.groupby("pizza_name")["order_id"]
              .count()
              .sort_values(ascending=False)
              .head(10)
        )
    top_pizzas_qty = {
        "names":  qty_by_pizza.index.tolist(),
        "values": qty_by_pizza.values.tolist(),
    }

    # ── hourly ──────────────────────────────────
    hourly_g = (
        df.groupby("hour")["order_id"]
          .nunique()
          .reindex(range(24), fill_value=0)
          .reset_index()
    )
    hourly = {
        "hours":  list(range(24)),
        "orders": hourly_g["order_id"].tolist(),
    }

    # ── day of week ─────────────────────────────
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_g = (
        df.groupby("dow")["total_price"]
          .sum()
          .reindex(range(7), fill_value=0)
    )
    dow = {
        "days":    dow_names,
        "revenue": [round(v, 2) for v in dow_g.values.tolist()],
    }

    # ── insights (auto-generated) ───────────────
    insights = generate_insights(summary, daily, monthly, dow, hourly,
                                  size, category, top_pizzas_rev, top_pizzas_qty)

    return {
        "summary":        summary,
        "daily":          daily,
        "weekly":         weekly,
        "monthly":        monthly,
        "category":       category,
        "size":           size,
        "top_pizzas_rev": top_pizzas_rev,
        "top_pizzas_qty": top_pizzas_qty,
        "hourly":         hourly,
        "dow":            dow,
        "insights":       insights,
    }


# ═══════════════════════════════════════════════
#  AUTO-INSIGHTS GENERATOR
# ═══════════════════════════════════════════════

def generate_insights(summary, daily, monthly, dow, hourly,
                       size, category, top_rev, top_qty) -> list:
    insights = []

    # 1. Top revenue driver
    if top_rev["names"]:
        top_name  = top_rev["names"][0].replace("The ", "")
        top_val   = top_rev["values"][0]
        pct_total = round(top_val / summary["total_revenue"] * 100, 1)
        insights.append({
            "icon": "🏆", "title": "Top Revenue Driver", "accent": "#f59e0b",
            "text": (f"<b>{top_name}</b> leads all {summary['num_pizza_types']} menu items "
                     f"with <b>${top_val:,.0f}</b> in revenue — {pct_total}% of total sales.")
        })

    # 2. Peak day
    best_dow_idx = int(np.argmax(dow["revenue"]))
    best_dow     = dow["days"][best_dow_idx]
    best_dow_rev = dow["revenue"][best_dow_idx]
    worst_dow_rev= min(dow["revenue"])
    drop_pct     = round((best_dow_rev - worst_dow_rev) / best_dow_rev * 100, 0)
    insights.append({
        "icon": "📅", "title": f"Peak Day: {best_dow}", "accent": "#ef4444",
        "text": (f"<b>{best_dow}</b> drives the highest daily revenue at "
                 f"<b>${best_dow_rev:,.0f}</b> annually — {int(drop_pct)}% above the slowest day.")
    })

    # 3. Lunch rush
    peak_hour     = hourly["hours"][int(np.argmax(hourly["orders"]))]
    peak_orders   = max(hourly["orders"])
    lunch_orders  = sum(hourly["orders"][11:15])
    total_orders_h= sum(hourly["orders"])
    lunch_pct     = round(lunch_orders / max(total_orders_h, 1) * 100, 0)
    hour_label    = f"{peak_hour % 12 or 12}{'am' if peak_hour < 12 else 'pm'}"
    insights.append({
        "icon": "⏰", "title": "Lunch Rush Dominates", "accent": "#14b8a6",
        "text": (f"The busiest hour is <b>{hour_label}</b> with <b>{peak_orders:,}</b> annual orders. "
                 f"The 11am–3pm window captures ~{int(lunch_pct)}% of daily traffic.")
    })

    # 4. Top size
    best_size_idx = int(np.argmax(size["revenue"]))
    best_size     = size["names"][best_size_idx]
    best_size_rev = size["revenue"][best_size_idx]
    size_pct      = round(best_size_rev / summary["total_revenue"] * 100, 0)
    insights.append({
        "icon": "📦", "title": f"{best_size}-Size Leads Revenue", "accent": "#8b5cf6",
        "text": (f"<b>{best_size}</b> size accounts for <b>${best_size_rev/1000:.0f}K</b> "
                 f"({int(size_pct)}% of revenue). Upselling to {best_size} is the biggest revenue lever.")
    })

    # 5. Best month
    best_mo_idx = int(np.argmax(monthly["revenue"]))
    worst_mo_idx= int(np.argmin(monthly["revenue"]))
    best_mo     = monthly["months"][best_mo_idx]
    worst_mo    = monthly["months"][worst_mo_idx]
    best_mo_rev = monthly["revenue"][best_mo_idx]
    worst_mo_rev= monthly["revenue"][worst_mo_idx]
    month_names = {
        "01":"January","02":"February","03":"March","04":"April",
        "05":"May","06":"June","07":"July","08":"August",
        "09":"September","10":"October","11":"November","12":"December"
    }
    best_mo_name  = month_names.get(best_mo.split("-")[1], best_mo)
    worst_mo_name = month_names.get(worst_mo.split("-")[1], worst_mo)
    insights.append({
        "icon": "📈", "title": "Best Month",  "accent": "#22c55e",
        "text": (f"<b>{best_mo_name}</b> was the strongest month at "
                 f"<b>${best_mo_rev:,.0f}</b>. Consistent seasonal strength worth leveraging.")
    })

    # 6. Seasonal dip
    insights.append({
        "icon": "📉", "title": "Seasonal Dip", "accent": "#fb923c",
        "text": (f"<b>{worst_mo_name}</b> posted the lowest revenue at "
                 f"<b>${worst_mo_rev:,.0f}</b>. A targeted promotion campaign could offset this trough.")
    })

    # 7. Category insight
    if category["names"]:
        top_cat     = category["names"][0]
        top_cat_rev = category["revenue"][0]
        top_cat_pct = round(top_cat_rev / summary["total_revenue"] * 100, 0)
        insights.append({
            "icon": "🍕", "title": f"{top_cat} Category Leads", "accent": "#38bdf8",
            "text": (f"<b>{top_cat}</b> pizzas account for <b>${top_cat_rev/1000:.0f}K</b> "
                     f"({int(top_cat_pct)}% of revenue). Menu diversity across categories is healthy.")
        })

    # 8. Concentration risk
    top5_rev = sum(top_rev["values"][:5])
    top5_pct = round(top5_rev / summary["total_revenue"] * 100, 0)
    insights.append({
        "icon": "💡", "title": "Revenue Concentration", "accent": "#ec4899",
        "text": (f"Top 5 pizzas represent <b>{int(top5_pct)}%</b> of total revenue from "
                 f"{summary['num_pizza_types']} items. Healthy spread but some concentration risk exists.")
    })

    # 9. Average order value
    insights.append({
        "icon": "💰", "title": "Average Order Value", "accent": "#fcd34d",
        "text": (f"Each order averages <b>${summary['avg_order_value']:.2f}</b>. "
                 f"Combo deals or upsell prompts could push this above the current baseline.")
    })

    return insights


# ═══════════════════════════════════════════════
#  NEURAL NETWORK (Pure NumPy — matches JS model)
# ═══════════════════════════════════════════════

class SimpleNN:
    """
    Single hidden-layer network:  input → tanh → linear
    Trained with SGD + momentum + learning-rate decay.
    Architecture mirrors the in-browser JS model.
    """
    def __init__(self, input_size: int, hidden_size: int, lr: float):
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        self.W1 = np.random.randn(hidden_size, input_size) * scale
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(1, hidden_size) * scale
        self.b2 = np.zeros(1)
        # Momentum buffers
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)
        self.lr       = lr
        self.momentum = 0.9

    def forward(self, X):
        """X: (batch, input_size)"""
        Z1 = X @ self.W1.T + self.b1        # (batch, hidden)
        H  = np.tanh(Z1)
        out = (H @ self.W2.T + self.b2).squeeze(-1)  # (batch,)
        return H, out

    def backward(self, X, H, out, Y):
        """Single-sample or mini-batch update. Returns MSE loss."""
        X = np.atleast_2d(X)
        H = np.atleast_2d(H)
        out = np.atleast_1d(out)
        Y   = np.atleast_1d(Y)
        batch = X.shape[0]

        d_out = (out - Y)                                # (batch,)
        dW2   = (d_out[:, None] * H).mean(0, keepdims=True)   # (1, hidden)
        db2   = d_out.mean(keepdims=True)
        dH    = d_out[:, None] * self.W2                # (batch, hidden)
        dZ1   = dH * (1 - H ** 2)                       # tanh deriv
        dW1   = (dZ1.T @ X) / batch                     # (hidden, input)
        db1   = dZ1.mean(0)                              # (hidden,)

        # Momentum SGD updates
        self.vW2 = self.momentum * self.vW2 + self.lr * dW2
        self.W2  -= self.vW2
        self.vb2 = self.momentum * self.vb2 + self.lr * db2
        self.b2  -= self.vb2

        self.vW1 = self.momentum * self.vW1 + self.lr * dW1
        self.W1  -= self.vW1
        self.vb1 = self.momentum * self.vb1 + self.lr * db1
        self.b1  -= self.vb1

        return float(np.mean(d_out ** 2))

    def predict(self, X):
        X = np.atleast_2d(X)
        _, out = self.forward(X)
        return out if out.shape[0] > 1 else float(out[0])


def run_forecast(daily_revenue: list, horizon: int = 60,
                 epochs: int = 200, lr: float = 0.005,
                 window: int = 14, hidden: int = 16) -> dict:
    """
    Train a simple neural net on normalized daily revenue,
    forecast `horizon` days, return full result payload.
    """
    rev = np.array(daily_revenue, dtype=np.float64)
    mn, mx = rev.min(), rev.max()

    # ── normalise ───────────────────────────────
    norm = (rev - mn) / (mx - mn + 1e-8)

    # ── build sequences ─────────────────────────
    X, Y = [], []
    for i in range(window, len(norm)):
        X.append(norm[i - window: i])
        Y.append(norm[i])
    X, Y = np.array(X), np.array(Y)

    split = int(len(X) * 0.85)
    trainX, trainY = X[:split], Y[:split]
    testX,  testY  = X[split:], Y[split:]

    np.random.seed(42)
    net = SimpleNN(window, hidden, lr)

    loss_history     = []
    val_loss_history = []

    # ── training loop ───────────────────────────
    for epoch in range(epochs):
        # Shuffle
        idx = np.random.permutation(len(trainX))
        H, out = net.forward(trainX[idx])
        epoch_loss = net.backward(trainX[idx], H, out, trainY[idx])
        loss_history.append(round(float(epoch_loss), 8))

        # Validation loss
        _, val_out = net.forward(testX)
        val_loss   = float(np.mean((val_out - testY) ** 2))
        val_loss_history.append(round(val_loss, 8))

        # LR decay every 100 epochs
        if epoch > 0 and epoch % 100 == 0:
            net.lr *= 0.8

    # ── test-set metrics ────────────────────────
    _, test_out = net.forward(testX)
    test_preds   = test_out * (mx - mn) + mn
    test_actuals = testY    * (mx - mn) + mn

    mae  = float(np.mean(np.abs(test_preds - test_actuals)))
    rmse = float(np.sqrt(np.mean((test_preds - test_actuals) ** 2)))
    mape = float(np.mean(np.abs((test_preds - test_actuals) /
                                 np.where(test_actuals == 0, 1e-8, test_actuals))) * 100)
    mean_act = test_actuals.mean()
    ss_tot   = float(np.sum((test_actuals - mean_act) ** 2))
    ss_res   = float(np.sum((test_actuals - test_preds) ** 2))
    r2       = round(1 - ss_res / (ss_tot + 1e-8), 4)
    residual_std = float(np.sqrt(np.mean((test_out - testY) ** 2)))

    # ── forecast ────────────────────────────────
    window_buf = norm[-window:].tolist()
    forecasts  = []
    for _ in range(horizon):
        p = float(net.predict(np.array(window_buf)))
        forecasts.append(p)
        window_buf = window_buf[1:] + [p]

    f_revenue = [round(float(v * (mx - mn) + mn), 2) for v in forecasts]
    avg_fc    = round(sum(f_revenue) / len(f_revenue), 2)
    total_fc  = round(sum(f_revenue), 2)

    # Confidence intervals
    ci_upper = [round(v + residual_std * (mx - mn) * (1 + i * 0.02), 2)
                for i, v in enumerate(f_revenue)]
    ci_lower = [round(max(0, v - residual_std * (mx - mn) * (1 + i * 0.02)), 2)
                for i, v in enumerate(f_revenue)]

    # Future dates
    # (frontend will supply last_date string)
    future_dates = []  # filled by caller

    return {
        "loss_history":     loss_history,
        "val_loss_history": val_loss_history,
        "test_preds":       [round(float(v), 2) for v in test_preds.tolist()],
        "test_actuals":     [round(float(v), 2) for v in test_actuals.tolist()],
        "forecast_revenue": f_revenue,
        "ci_upper":         ci_upper,
        "ci_lower":         ci_lower,
        "metrics": {
            "mae":       round(mae, 2),
            "rmse":      round(rmse, 2),
            "mape":      round(mape, 2),
            "r2":        r2,
            "avg_fc":    avg_fc,
            "total_fc":  total_fc,
        },
    }


# ═══════════════════════════════════════════════
#  FLASK ROUTES
# ═══════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "PizzaIQ API"})


@app.route("/upload", methods=["POST"])
def upload():
    """
    Accept a CSV file upload, run the full analytics pipeline,
    and return the computed data payload as JSON.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Send as multipart/form-data key 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        content = file.read().decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(content))
        analytics = process_csv(df)
        return jsonify({"success": True, "data": analytics})
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500


@app.route("/forecast", methods=["POST"])
def forecast():
    """
    Train neural net + return forecast.
    Body (JSON):
      {
        "daily_revenue": [...],   # list of floats
        "last_date":     "YYYY-MM-DD",
        "horizon":       60,      # optional, default 60
        "epochs":        200,     # optional, default 200
        "lr":            0.005    # optional, default 0.005
      }
    """
    body = request.get_json(force=True, silent=True) or {}

    daily_revenue = body.get("daily_revenue")
    if not daily_revenue or not isinstance(daily_revenue, list):
        return jsonify({"error": "'daily_revenue' list is required."}), 400

    horizon   = int(body.get("horizon", 60))
    epochs    = min(int(body.get("epochs", 200)), 1000)   # cap at 1000 for safety
    lr        = float(body.get("lr", 0.005))
    last_date = body.get("last_date", "")

    try:
        result = run_forecast(daily_revenue, horizon=horizon,
                              epochs=epochs, lr=lr)

        # Generate future dates server-side
        if last_date:
            try:
                base = datetime.strptime(last_date, "%Y-%m-%d")
                result["future_dates"] = [
                    (base + timedelta(days=i + 1)).strftime("%Y-%m-%d")
                    for i in range(horizon)
                ]
            except ValueError:
                result["future_dates"] = []
        else:
            result["future_dates"] = []

        return jsonify({"success": True, "data": result})

    except Exception as e:
        return jsonify({"error": f"Forecast error: {str(e)}"}), 500


# ═══════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    # print("╔══════════════════════════════════════╗")
    # print("║  🍕  PizzaIQ Analytics API           ║")
    # print("║  Running at http://localhost:5000    ║")
    # print("║  POST /upload  — CSV analytics       ║")
    # print("║  POST /forecast — ML forecasting     ║")
    # print("╚══════════════════════════════════════╝")
    app.run(host="0.0.0.0", port=5000, debug=False)