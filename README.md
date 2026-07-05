# Flight Search & Analytics Platform

An interactive Streamlit app for searching flights, exploring analytics, predicting prices with machine learning, and querying flight data in plain English — all backed by a MySQL database.

## ✨ Features

- **Check Flights** — Search flights by source and destination, with filters for max price and non-stop only, plus sorting by price or duration.
- **CSV Export** — Download any search result set directly as a CSV file.
- **Analytics Dashboard** — Flights per airline (pie chart), busiest airports (bar chart), daily flight volume (line chart), and price spread by airline (box plot).
- **Price Predictor** — A RandomForest regression model trained on historical fares predicts ticket price from airline, route, stops, and duration, with live model accuracy (MAE) displayed.
- **Ask in Plain English** — Type a question like *"fastest IndiGo flight from Bangalore to Kolkata"* or *"flights under 5000 from Chennai to Hyderabad"* and the Groq API (Llama 3.3 70B) extracts source, destination, price cap, airline, non-stop preference, and sort intent (cheapest vs. fastest), then runs the search. Includes clickable example queries, a "Detected filters" line showing exactly what was parsed, an AI-generated one-line summary of the results, and a session history of your last few questions.
- **Designed Landing Page** — A styled hero section with live stats (flights indexed, airlines covered, cities served) pulled straight from the database.

## 🛠️ Tech Stack

- **Language:** Python
- **Framework:** Streamlit
- **Database:** MySQL (via `mysql-connector-python`)
- **Data Processing:** Pandas
- **Machine Learning:** scikit-learn (RandomForestRegressor)
- **Visualization:** Plotly (`plotly.express`, `plotly.graph_objs`)
- **Natural Language Interface:** Groq API (`groq` SDK, Llama 3.3 70B)
- **Config Management:** `python-dotenv` (`.env` file)
- **Package Management:** `pip` (`requirements.txt`)

## 📁 Project Structure

```
flights-sql-dashboard/
├── app.py              # Main Streamlit app — UI, charts, ML, and NL search logic
├── dbhelper.py         # DB class — all MySQL queries, kept separate from UI code
├── requirements.txt    # Project dependencies
├── .env                # Local environment variables (DB + API credentials, not committed)
└── .gitignore
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- MySQL server running locally (e.g. via MySQL Workbench) with a `flights` database and `flights` table
- A free Groq API key from [console.groq.com](https://console.groq.com) (only needed for "Ask in Plain English")

### Installation

```bash
git clone https://github.com/ai-shyamal-rai/flights-sql-dashboard.git
cd flights-sql-dashboard
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```
DB_HOST=127.0.0.1
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_NAME=flights
GROQ_API_KEY=your_groq_api_key
```

### Running the App

```bash
streamlit run app.py
```

The app will open automatically in your browser at: **`http://localhost:8501`**

## 📊 Usage

1. Launch the app.
2. In the sidebar, choose a menu option: **Check Flights**, **Analytics**, **Price Predictor**, or **Ask in Plain English**.
3. **Check Flights** — pick a source and destination, set filters, and click "Search Flights"; download results as CSV if needed.
4. **Analytics** — view airline, airport, and pricing trends across the full dataset.
5. **Price Predictor** — select an airline, route, stop count, and duration, then click "Predict price" for an estimated fare.
6. **Ask in Plain English** — click an example query or type your own natural language flight request; review the "Detected" filters line, read the AI summary of results, and check "Previous questions this session" to revisit earlier searches.

## 🗺️ Roadmap

- [ ] Add user accounts and saved searches
- [ ] Deploy with a managed MySQL instance (e.g. PlanetScale, RDS) and Streamlit Community Cloud
- [ ] Add a proper train/test evaluation page and model versioning for the price predictor
- [ ] Support multi-city and round-trip search

## 📄 License

This project is licensed under the MIT License — feel free to use and modify it.

## 🙋 Author

Built by [Shyamal Rai](https://github.com/ai-shyamal-rai).