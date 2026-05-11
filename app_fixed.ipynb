"""
Bank Stress Testing Streamlit App
==================================
Front-end dashboard for visualizing stress test scenarios and loss forecasts.
Models are pre-trained; no retraining occurs in this app.

Author: Generated with Claude Code
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

MODELS_DIR = Path("./models")
CAPITAL_ADEQUACY_THRESHOLD = 6000  # $M - MODIFY THIS VALUE AS NEEDED

PORTFOLIO_MAP = {
    1: "Residential Mortgages",
    2: "Commercial Loans",
    3: "Credit Cards",
    4: "Securities Portfolio"
}

SCENARIO_DESCRIPTIONS = {
    "Economic_Recession_Severe": """
    **Economic Recession (Severe)**
    - GDP decline: -5.0%
    - Unemployment spike: +8.0%
    - Housing prices: -20.0%
    - Credit spreads: +400 bps
    - VIX: 60 (extreme fear)
    """,
    "Interest_Rate_Shock_Moderate": """
    **Interest Rate Shock (Moderate)**
    - Fed Funds Rate: +300 bps
    - Treasury 10Y: +250 bps
    - GDP impact: -1.5%
    - Housing prices: -8.0%
    - VIX: 30 (elevated)
    """,
    "Market_Volatility_Crisis_Severe": """
    **Market Volatility Crisis (Severe)**
    - VIX: 75 (panic)
    - Credit spreads: +500 bps
    - GDP: -3.0%
    - Unemployment: +5.0%
    - Housing prices: -15.0%
    """,
    "Mild_Stress_Baseline": """
    **Mild Stress (Baseline)**
    - GDP slowdown: -1.0%
    - Unemployment: +2.0%
    - Housing prices: -5.0%
    - Credit spreads: +100 bps
    - VIX: 25 (moderate concern)
    """
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_data
def load_metadata():
    """Load model metadata (AIC, BIC, ARIMA orders, balances)."""
    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Convert string keys to int
    return {int(k): v for k, v in metadata.items()}


@st.cache_data
def load_forecasts():
    """Load precomputed stress test forecasts from CSV."""
    forecast_path = MODELS_DIR / "stress_test_forecasts.csv"
    df = pd.read_csv(forecast_path)
    df['forecast_date'] = pd.to_datetime(df['forecast_date'])
    return df

@st.cache_data
def load_historical_loss_rates():
    """
    Load historical loss rates per portfolio.

    If the CSV doesn't exist, it will be generated from the database.
    This ensures the app works even if historical_loss_rates.csv is missing.
    """
    hist_path = MODELS_DIR / "historical_loss_rates.csv"

    # If file doesn't exist, generate it from database
    if not hist_path.exists():
        try:
            # Import database dependencies
            import sys
            import os
            sys.path.insert(0, str(Path(__file__).parent))
            from dotenv import load_dotenv
            from PostgresAgent import PostgresAgent

            # Load environment variables
            load_dotenv()

            # Get username from environment (USER env var, not POSTGRES_USER)
            username = os.environ.get('USER', 'jackn')

            # Initialize database agent
            agent = PostgresAgent(
                username=os.getenv('POSTGRES_USER'),
                password=os.getenv('POSTGRES_PASSWD'),
                host=os.getenv('POSTGRES_HOST'),
                port=int(os.getenv('POSTGRES_PORT', 5432)),
                database=os.getenv('POSTGRES_DATABASE')
            )

            # Load historical loan losses from database (same as modeling.ipynb)
            df_losses = agent.execute_dml(f"SELECT * FROM {username}.historical_loan_losses ORDER BY cal_date, portfolio_id;")

            # Convert dates to datetime
            df_losses['cal_date'] = pd.to_datetime(df_losses['cal_date'])

            # Map portfolio IDs to names (same mapping as modeling.ipynb)
            portfolio_map = {
                1: 'Residential_Mortgages',
                2: 'Commercial_Loans',
                3: 'Credit_Cards',
                4: 'Securities_Portfolio'
            }
            df_losses['portfolio_name'] = df_losses['portfolio_id'].map(portfolio_map)

            # Select only the columns needed for the app
            df_historical = df_losses[['cal_date', 'portfolio_id', 'portfolio_name', 'loss_rate_percent']].copy()

            # Save to CSV
            MODELS_DIR.mkdir(exist_ok=True)
            df_historical.to_csv(hist_path, index=False)

            st.info(f"✅ Generated {hist_path.name} from database")

        except Exception as e:
            st.error(f"""
            **Error: Could not load historical loss rates.**

            The file `{hist_path}` is missing and could not be generated from the database.

            **Error details:** {str(e)}

            **To fix this issue:**
            1. Ensure you have run `modeling.ipynb` to generate all required files, OR
            2. Check that your database connection is configured in `.env` file
            """)
            # Return empty dataframe to prevent crash
            return pd.DataFrame(columns=['cal_date', 'portfolio_id', 'portfolio_name', 'loss_rate_percent'])

    # Load the CSV file
    df = pd.read_csv(hist_path)
    df["cal_date"] = pd.to_datetime(df["cal_date"])
    return df


@st.cache_resource
def load_arimax_models():
    """
    Load trained ARIMAX models for real-time forecasting.

    Returns None if joblib is not installed, models are missing, or any error occurs.
    This makes real-time forecasting optional without breaking precomputed mode.
    """
    try:
        # Import joblib only when needed (not at top-level)
        import joblib

        models = {}
        metadata = load_metadata()

        for portfolio_id in [1, 2, 3, 4]:
            # Use PORTFOLIO_MAP to get consistent naming
            portfolio_display_name = PORTFOLIO_MAP[portfolio_id]
            # Convert to filename format: "Residential Mortgages" -> "residential_mortgages"
            portfolio_filename = portfolio_display_name.lower().replace(" ", "_")
            model_path = MODELS_DIR / f"arimax_{portfolio_filename}.pkl"

            if not model_path.exists():
                return None

            models[portfolio_id] = joblib.load(model_path)

        return models
    except ImportError:
        # joblib not installed
        return None
    except Exception:
        # Any other error (missing metadata keys, corrupt files, etc.)
        return None


def get_scenario_data(scenario_name, df_forecasts):
    """Filter forecasts for a specific scenario."""
    return df_forecasts[df_forecasts['scenario_name'] == scenario_name].copy()


def compute_capital_adequacy(total_loss, threshold=CAPITAL_ADEQUACY_THRESHOLD):
    """
    Determine capital adequacy status.

    Args:
        total_loss: Total 12-month loss in $M
        threshold: Maximum acceptable loss in $M

    Returns:
        tuple: (status, color) where status is 'PASS' or 'FAIL'
    """
    if total_loss <= threshold:
        return "PASS", "#2ECC71"  # Green
    else:
        return "FAIL", "#E74C3C"  # Red


def get_scenario_description(scenario_name):
    """Get scenario description text."""
    return SCENARIO_DESCRIPTIONS.get(scenario_name, "No description available.")


def create_bar_chart(scenario_df):
    """Create bar chart showing total losses by portfolio."""
    # Aggregate total losses by portfolio
    portfolio_losses = scenario_df.groupby(['portfolio_id', 'portfolio_name'])['loss_amount_millions'].sum().reset_index()
    portfolio_losses = portfolio_losses.sort_values('portfolio_id')

    fig = go.Figure()

    colors = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12']

    fig.add_trace(go.Bar(
        x=portfolio_losses['portfolio_name'],
        y=portfolio_losses['loss_amount_millions'],
        marker=dict(color=colors),
        text=portfolio_losses['loss_amount_millions'].round(1),
        texttemplate='$%{text}M',
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Total Loss: $%{y:.2f}M<extra></extra>'
    ))

    fig.update_layout(
        title="Total 12-Month Losses by Portfolio",
        xaxis_title="Portfolio",
        yaxis_title="Total Loss ($M)",
        showlegend=False,
        height=400,
        template="plotly_white",
        hovermode='x'
    )

    return fig

def create_forecast_line_chart(scenario_df, hist_df):
    """Create line chart showing historical and forecasted loss rates for all portfolios."""
    fig = go.Figure()

    colors = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12']

    for idx, portfolio_id in enumerate(sorted(scenario_df['portfolio_id'].unique())):
        portfolio_name = PORTFOLIO_MAP[portfolio_id]

        # Historical data for this portfolio
        hist_data = (
            hist_df[hist_df["portfolio_id"] == portfolio_id]
            .sort_values("cal_date")
        )

        # Forecast data for this portfolio
        forecast_data = (
            scenario_df[scenario_df['portfolio_id'] == portfolio_id]
            .sort_values('forecast_date')
        )

        # Historical line
        fig.add_trace(go.Scatter(
            x=hist_data["cal_date"],
            y=hist_data["loss_rate_percent"],
            mode="lines",
            name=f"{portfolio_name} (Historical)",
            line=dict(color=colors[idx], width=2, dash="solid"),
            hovertemplate='<b>%{fullData.name}</b><br>Date: %{x|%b %Y}<br>Loss Rate: %{y:.2f}%<extra></extra>'
        ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast_data["forecast_date"],
            y=forecast_data["loss_rate_percent"],
            mode="lines+markers",
            name=f"{portfolio_name} (Forecast)",
            line=dict(color=colors[idx], width=2, dash="dash"),
            marker=dict(size=6),
            hovertemplate='<b>%{fullData.name}</b><br>Date: %{x|%b %Y}<br>Loss Rate: %{y:.2f}%<extra></extra>'
        ))

    fig.update_layout(
        title="Historical and 12-Month Forecasted Loss Rates",
        xaxis_title="Month",
        yaxis_title="Loss Rate (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        template="plotly_white",
        hovermode='x unified'
    )

    return fig



def create_capital_gauge(total_loss, threshold=CAPITAL_ADEQUACY_THRESHOLD):
    """Create gauge chart for capital adequacy."""
    status, color = compute_capital_adequacy(total_loss, threshold)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=total_loss,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Capital Adequacy: {status}", 'font': {'size': 20, 'color': color}},
        delta={'reference': threshold, 'increasing': {'color': "#E74C3C"}, 'decreasing': {'color': "#2ECC71"}},
        gauge={
            'axis': {'range': [None, threshold * 1.5], 'tickformat': '$,.0f'},
            'bar': {'color': color},
            'steps': [
                {'range': [0, threshold], 'color': '#D5F5E3'},
                {'range': [threshold, threshold * 1.5], 'color': '#FADBD8'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        },
        number={'suffix': 'M', 'prefix': '$'}
    ))

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=80, b=20)
    )

    return fig


def create_summary_table(scenario_df, metadata):
    """Create summary table with model details and losses."""
    summary_data = []

    for portfolio_id in sorted(scenario_df['portfolio_id'].unique()):
        portfolio_data = scenario_df[scenario_df['portfolio_id'] == portfolio_id]
        meta = metadata[portfolio_id]

        total_loss = portfolio_data['loss_amount_millions'].sum()
        avg_loss_rate = portfolio_data['loss_rate_percent'].mean()

        arima_order = tuple(meta['order'])

        summary_data.append({
            'Portfolio': PORTFOLIO_MAP[portfolio_id],
            'ARIMA Order': f"{arima_order}",
            'AIC': f"{meta['aic']:.2f}",
            'BIC': f"{meta['bic']:.2f}",
            'Avg Loss Rate (%)': f"{avg_loss_rate:.2f}",
            'Total Loss ($M)': f"{total_loss:.2f}",
            'Balance ($M)': f"{meta['balance_millions']:.0f}"
        })

    df_summary = pd.DataFrame(summary_data)
    return df_summary


def generate_real_time_forecast(models, metadata, economic_inputs, n_periods=12):
    """
    Generate real-time forecasts using ARIMAX models and custom economic inputs.

    Args:
        models: Dictionary of loaded ARIMAX models by portfolio_id
        metadata: Model metadata including predictors and balances
        economic_inputs: Dictionary of economic variable values
        n_periods: Number of months to forecast (default 12)

    Returns:
        DataFrame with forecasts in same format as precomputed scenarios
    """
    forecast_data = []
    hist_df = load_historical_loss_rates()

    # Get the last historical date to continue from
    last_hist_date = hist_df['cal_date'].max()

    for portfolio_id, model in models.items():
        meta = metadata[portfolio_id]
        predictors = meta['predictors']
        balance = meta['balance_millions']
        portfolio_name = PORTFOLIO_MAP[portfolio_id]

        # Build exogenous variable array
        X_values = [economic_inputs[pred] for pred in predictors]
        X_scenario = np.tile(X_values, (n_periods, 1))

        # Generate forecast with confidence intervals
        try:
            forecast_result = model.predict(
                n_periods=n_periods,
                X=X_scenario,
                return_conf_int=True,
                alpha=0.05
            )

            loss_rates = forecast_result[0]
            conf_int = forecast_result[1]

        except Exception:
            # Fallback if confidence intervals not supported
            loss_rates = model.predict(n_periods=n_periods, X=X_scenario)
            conf_int = None

        # Convert to dollar amounts
        loss_amounts = (loss_rates / 100) * balance

        # Generate forecast dates
        for i in range(n_periods):
            forecast_date = last_hist_date + pd.DateOffset(months=i+1)

            forecast_data.append({
                'scenario_name': 'Real_Time_Custom',
                'portfolio_id': portfolio_id,
                'portfolio_name': portfolio_name,
                'forecast_date': forecast_date,
                'loss_rate_percent': loss_rates[i],
                'loss_amount_millions': loss_amounts[i],
                'lower_ci_percent': conf_int[i][0] if conf_int is not None else loss_rates[i] * 0.9,
                'upper_ci_percent': conf_int[i][1] if conf_int is not None else loss_rates[i] * 1.1
            })

    return pd.DataFrame(forecast_data)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Page configuration
    st.set_page_config(
        page_title="Bank Stress Testing Dashboard",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Title
    st.title("🏦 Bank Stress Testing Dashboard")
    st.markdown("**12-Month Loss Forecasting for Economic Stress Scenarios**")
    st.markdown("---")

    # Load data (common to both modes)
    metadata = load_metadata()
    df_forecasts = load_forecasts()
    hist_df = load_historical_loss_rates()

    # Get available scenarios
    scenarios = sorted(df_forecasts['scenario_name'].unique())

    # =========================================================================
    # SIDEBAR: MODE SELECTION
    # =========================================================================
    st.sidebar.header("🎯 Forecast Mode")
    forecast_mode = st.sidebar.radio(
        "Select Mode:",
        ["Precomputed Scenarios", "Real-Time Forecast"],
        index=0,
        help="Choose between predefined scenarios or create custom forecasts with real-time inputs"
    )

    st.sidebar.markdown("---")

    # =========================================================================
    # MODE 1: PRECOMPUTED SCENARIOS (EXISTING BEHAVIOR - UNCHANGED)
    # =========================================================================
    if forecast_mode == "Precomputed Scenarios":
        st.sidebar.header("⚙️ Scenario Selection")
        selected_scenario = st.sidebar.selectbox(
            "Select Economic Stress Scenario:",
            scenarios,
            format_func=lambda x: x.replace('_', ' ').title()
        )

        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Capital Adequacy Threshold:**  \n${CAPITAL_ADEQUACY_THRESHOLD:,.0f}M")
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Portfolios:**")
        for pid, pname in PORTFOLIO_MAP.items():
            balance = metadata[pid]['balance_millions']
            st.sidebar.markdown(f"- {pname}: ${balance:.0f}M")

        # Get scenario data
        scenario_df = get_scenario_data(selected_scenario, df_forecasts)
        scenario_title = selected_scenario.replace('_', ' ').title()
        scenario_description = get_scenario_description(selected_scenario)

    # =========================================================================
    # MODE 2: REAL-TIME FORECAST (NEW ADDITIVE FEATURE)
    # =========================================================================
    else:
        # Try to load ARIMAX models
        arimax_models = load_arimax_models()

        if arimax_models is None:
            # Models not available - show message and exit gracefully
            st.info("🔧 **Real-Time Forecast Mode**")
            st.warning("""
            **Real-time forecasting is not available with the current repository artifacts.**

            Please use **Precomputed Scenarios** mode to view stress test results.

            **To enable real-time forecasting, ensure:**
            - `joblib` package is installed (`pip install joblib`)
            - All ARIMAX model files exist in the `models/` directory:
              - `arimax_residential_mortgages.pkl`
              - `arimax_commercial_loans.pkl`
              - `arimax_credit_cards.pkl`
              - `arimax_securities_portfolio.pkl`
            - `model_metadata.json` contains `predictors` and `balance_millions` keys for each portfolio
            """)

            st.sidebar.markdown(f"**Capital Adequacy Threshold:**  \n${CAPITAL_ADEQUACY_THRESHOLD:,.0f}M")
            st.sidebar.markdown("---")
            st.sidebar.markdown("**Note:** Switch to Precomputed Scenarios mode to view forecasts.")

            return

        # Models available - show economic input form
        st.sidebar.header("📊 Economic Variables")
        st.sidebar.markdown("*Adjust sliders to create custom scenario*")
        st.sidebar.markdown("---")

        # Economic inputs
        economic_inputs = {}

        st.sidebar.markdown("**Macro Economic**")
        economic_inputs['gdp_growth'] = st.sidebar.slider(
            "GDP Growth (%)",
            min_value=-8.0,
            max_value=5.0,
            value=-2.0,
            step=0.5,
            help="Annual GDP growth rate"
        )

        economic_inputs['unemployment_rate'] = st.sidebar.slider(
            "Unemployment Rate (%)",
            min_value=3.0,
            max_value=15.0,
            value=6.5,
            step=0.5,
            help="National unemployment rate"
        )

        st.sidebar.markdown("**Housing & Credit**")
        economic_inputs['housing_price_change'] = st.sidebar.slider(
            "Housing Price Change (%)",
            min_value=-30.0,
            max_value=15.0,
            value=-10.0,
            step=1.0,
            help="Year-over-year housing price change"
        )

        economic_inputs['credit_spread_bps'] = st.sidebar.slider(
            "Credit Spread (bps)",
            min_value=50,
            max_value=800,
            step=25,
            value=300,
            help="Credit spread over risk-free rate (basis points)"
        )

        st.sidebar.markdown("**Financial Markets**")
        economic_inputs['fed_funds_rate'] = st.sidebar.slider(
            "Fed Funds Rate (%)",
            min_value=0.0,
            max_value=8.0,
            value=3.5,
            step=0.25,
            help="Federal Reserve target rate"
        )

        economic_inputs['vix_level'] = st.sidebar.slider(
            "VIX Level",
            min_value=10,
            max_value=85,
            step=5,
            value=40,
            help="CBOE Volatility Index (market fear gauge)"
        )

        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Capital Adequacy Threshold:**  \n${CAPITAL_ADEQUACY_THRESHOLD:,.0f}M")
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Portfolios:**")
        for pid, pname in PORTFOLIO_MAP.items():
            balance = metadata[pid]['balance_millions']
            st.sidebar.markdown(f"- {pname}: ${balance:.0f}M")

        # Generate real-time forecast
        with st.spinner("Generating real-time forecast..."):
            scenario_df = generate_real_time_forecast(arimax_models, metadata, economic_inputs)

        scenario_title = "Real-Time Custom Forecast"
        scenario_description = f"""
        **Custom Economic Scenario (Real-Time)**
        - GDP Growth: {economic_inputs['gdp_growth']:.1f}%
        - Unemployment Rate: {economic_inputs['unemployment_rate']:.1f}%
        - Housing Price Change: {economic_inputs['housing_price_change']:.1f}%
        - Credit Spread: {economic_inputs['credit_spread_bps']} bps
        - Fed Funds Rate: {economic_inputs['fed_funds_rate']:.2f}%
        - VIX Level: {economic_inputs['vix_level']}

        *Note: Economic variables are held constant over the 12-month forecast period.*
        """

    # =========================================================================
    # COMMON DISPLAY LOGIC (REUSES EXISTING LAYOUT FOR BOTH MODES)
    # =========================================================================

    # Calculate total loss
    total_loss = scenario_df['loss_amount_millions'].sum()
    status, color = compute_capital_adequacy(total_loss)

    # Display selected scenario
    st.header(f"📊 Scenario: {scenario_title}")

    # Key metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Total 12-Month Loss",
            value=f"${total_loss:,.0f}M",
            delta=f"{total_loss - CAPITAL_ADEQUACY_THRESHOLD:,.0f}M vs Threshold",
            delta_color="inverse"
        )

    with col2:
        st.metric(
            label="Capital Adequacy Status",
            value=status,
            delta=None
        )
        if status == "PASS":
            st.success("✅ Within acceptable loss threshold")
        else:
            st.error("❌ Exceeds acceptable loss threshold")

    with col3:
        st.metric(
            label="Number of Portfolios",
            value=len(PORTFOLIO_MAP),
            delta=None
        )

    st.markdown("---")

    # 2×2 Visualization Grid
    st.subheader("📈 Visualizations")

    # Row 1
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            create_bar_chart(scenario_df),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            create_forecast_line_chart(scenario_df, hist_df),
            use_container_width=True
        )

    # Row 2
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            create_capital_gauge(total_loss),
            use_container_width=True
        )

    with col2:
        st.subheader("Summary Table")
        summary_df = create_summary_table(scenario_df, metadata)
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            height=400
        )

    # Footer with scenario description
    st.markdown("---")
    st.subheader("📋 Scenario Economic Assumptions")
    st.markdown(scenario_description)

    # Additional info
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
    <p><strong>Bank Stress Testing Dashboard</strong> | Powered by Auto-ARIMAX Models</p>
    <p>Models trained on 120 months of historical data (2014-2023) | Forecasts generated for 12 months ahead (2024-2025)</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
