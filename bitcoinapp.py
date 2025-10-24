import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("📈 Bitcoin Price Prediction App")
st.write("This app uses an **LSTM model** to predict Bitcoin prices using Yahoo Finance data.")

# User inputs
start_date = st.date_input("Select start date", pd.to_datetime("2015-01-01"))
end_date = st.date_input("Select end date", pd.to_datetime("today"))

# ------------------------------
# Load Bitcoin Data
# ------------------------------
st.subheader("1️⃣ Bitcoin Historical Data")
df = yf.download("BTC-USD", start=start_date, end=end_date)
st.write(df.tail())

# Plot closing price
st.line_chart(df["Close"])

# ------------------------------
# Preprocessing
# ------------------------------
data = df[["Close"]].values
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

time_step = 60

X, y = [], []
for i in range(time_step, len(scaled_data)):
    X.append(scaled_data[i-time_step:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ------------------------------
# LSTM Model
# ------------------------------
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

# ------------------------------
# Predictions
# ------------------------------
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions.reshape(-1,1))
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1,1))

# Plot Predictions vs Actual
st.subheader("2️⃣ Model Predictions vs Actual Prices")
plt.figure(figsize=(10,6))
plt.plot(y_test_rescaled, label="Actual Price")
plt.plot(predictions, label="Predicted Price")
plt.legend()
st.pyplot(plt)

# ------------------------------
# Performance Metrics
# ------------------------------
mse = mean_squared_error(y_test_rescaled, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_rescaled, predictions)
r2 = r2_score(y_test_rescaled, predictions)

st.subheader("3️⃣ Model Performance")
st.write(f"✅ RMSE: {rmse:.2f}")
st.write(f"✅ MAE: {mae:.2f}")
st.write(f"✅ R² Score: {r2:.2f}")

# ------------------------------
# Future Prediction
# ------------------------------
st.subheader("4️⃣ Future 30-Day Prediction")

last_data = scaled_data[-time_step:]
future_input = last_data.reshape(1, time_step, 1)

future_preds = []
for _ in range(30):
    pred = model.predict(future_input, verbose=0)
    value = pred[0][0]   # get scalar
    future_preds.append(value)

    # update input correctly
    future_input = np.append(future_input[:, 1:, :], [[[value]]], axis=1)

# inverse transform predictions
future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

# generate future dates
future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=30)

# create dataframe for plotting
future_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Price": future_preds.flatten()
})
future_df.set_index("Date", inplace=True)

# plot
st.line_chart(future_df)


# Download predictions
future_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_preds.flatten()})
csv = future_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Predictions", data=csv, file_name="future_bitcoin_predictions.csv", mime="text/csv")


