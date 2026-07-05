# uv pip install streamlit pandas scikit-learn plotly

# uv run streamlit run app.py - run the app


# FRONTEND FILE

import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import os
from dbhelper import DB

# ---------- caching the DB connection so it isn't re-opened on every rerun ----------
@st.cache_resource
def get_db():
    return DB()

db = get_db()

# --- NEW: stop early with a clear message if the DB connection failed,
# instead of crashing later inside individual query calls ---
if db.mycursor is None:
    st.error(
        f"Couldn't connect to the database.\n\n"
        f"**Error:** {db.connection_error}\n\n"
        f"Check that MySQL is running and that DB_HOST / DB_USER / DB_PASSWORD / "
        f"DB_NAME are set correctly (or unset, to use the localhost/root/no-password defaults)."
    )
    st.stop()

st.sidebar.title('Flights Analytics')

user_option = st.sidebar.selectbox(
    'Menu', ['Select One', 'Check Flights', 'Analytics', 'Price Predictor', 'Ask in Plain English']
)

# =========================================================
# 1. CHECK FLIGHTS  (with filters + sort + CSV export)
# =========================================================
if user_option == 'Check Flights':
    st.title('Check Flights')

    col1, col2 = st.columns(2)
    city = db.fetch_city_names()
    with col1:
        source = st.selectbox('Source', sorted(city))
    with col2:
        destination = st.selectbox('Destination', sorted(city))

    # --- NEW: filter controls ---
    with st.expander("Filters"):
        f1, f2, f3 = st.columns(3)
        with f1:
            max_price = st.slider('Max price (₹)', 0, 50000, 50000, step=500)
        with f2:
            non_stop_only = st.checkbox('Non-stop only')
        with f3:
            sort_by = st.selectbox('Sort by', ['Price', 'Duration'])

    if st.button('Search Flights'):
        if source == destination:
            st.warning('Source and destination cannot be the same.')
        else:
            results = db.fetch_all_flights_filtered(
                source, destination,
                max_price=max_price,
                non_stop_only=non_stop_only,
                sort_by=sort_by
            )

            if results.empty:
                # NOTE: original app just showed an empty dataframe here with
                # no explanation — bad UX when filters are too strict.
                st.info('No flights match these filters. Try relaxing the price cap or stops.')
            else:
                st.success(f'{len(results)} flights found')
                st.dataframe(results, use_container_width=True)

                # --- NEW: CSV export ---
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    'Download results as CSV',
                    data=csv,
                    file_name=f'{source}_to_{destination}_flights.csv',
                    mime='text/csv'
                )

# =========================================================
# 2. ANALYTICS  (original 3 charts + 1 new box plot)
# =========================================================
elif user_option == 'Analytics':

    airline, frequency = db.fetch_airline_frequency()
    fig = go.Figure(go.Pie(labels=airline, values=frequency,
                            hoverinfo='label+percent', textinfo='value'))
    st.header('Flights per Airline')
    st.plotly_chart(fig)

    city, frequency1 = db.busy_airport()
    fig = px.bar(x=city, y=frequency1)
    st.header('Busiest Airports')
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)

    date, frequency2 = db.daily_frequency()
    fig = px.line(x=date, y=frequency2)
    st.header('Flights per Day')
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)

    # --- NEW: price spread per airline ---
    price_df = db.fetch_price_by_airline()
    fig = px.box(price_df, x='Airline', y='Price')
    st.header('Price Spread by Airline')
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)

# =========================================================
# 3. PRICE PREDICTOR  (NEW — shows ML, not just API calls)
# =========================================================
elif user_option == 'Price Predictor':
    st.title('Flight Price Predictor')
    st.caption('A simple regression model trained on the historical flights table.')

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    @st.cache_resource
    def train_model():
        df = db.fetch_training_data()

        # Duration is usually stored like "2h 30m" in this dataset — convert to minutes
        def to_minutes(x):
            h, m = 0, 0
            for part in str(x).split():
                if 'h' in part:
                    h = int(part.replace('h', '') or 0)
                elif 'm' in part:
                    m = int(part.replace('m', '') or 0)
            return h * 60 + m

        df['Duration_mins'] = df['Duration'].apply(to_minutes)
        df['Total_Stops'] = pd.to_numeric(df['Total_Stops'], errors='coerce').fillna(0)

        X = df[['Airline', 'Source', 'Destination', 'Total_Stops', 'Duration_mins']]
        y = df['Price']

        preprocess = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Airline', 'Source', 'Destination'])
        ], remainder='passthrough')

        model = Pipeline([
            ('prep', preprocess),
            ('rf', RandomForestRegressor(n_estimators=200, random_state=42))
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, model.predict(X_test))
        return model, mae, sorted(df['Airline'].unique()), sorted(df['Source'].unique()), sorted(df['Destination'].unique())

    model, mae, airlines, sources, destinations = train_model()
    st.caption(f'Model mean absolute error on held-out data: ₹{mae:,.0f}')

    c1, c2, c3 = st.columns(3)
    with c1:
        p_airline = st.selectbox('Airline', airlines)
    with c2:
        p_source = st.selectbox('Source', sources, key='p_source')
    with c3:
        p_dest = st.selectbox('Destination', destinations, key='p_dest')

    stops = st.slider('Total stops', 0, 3, 0)
    duration_mins = st.slider('Duration (minutes)', 30, 1500, 120)

    if st.button('Predict price'):
        input_df = pd.DataFrame([{
            'Airline': p_airline, 'Source': p_source, 'Destination': p_dest,
            'Total_Stops': stops, 'Duration_mins': duration_mins
        }])
        prediction = model.predict(input_df)[0]
        st.metric('Predicted price', f'₹{prediction:,.0f}')

# =========================================================
# 4. ASK IN PLAIN ENGLISH  (NEW — LLM-powered natural language filter)
# =========================================================
elif user_option == 'Ask in Plain English':
    st.title('Ask in Plain English')
    st.caption('Ask for what you want, the way you\'d ask a friend — Groq does the parsing.')

    # --- NEW: example query chips so people don't stare at a blank box ---
    example_queries = [
        "cheapest non-stop flight from Delhi to Mumbai",
        "fastest IndiGo flight from Bangalore to Kolkata",
        "flights under 5000 from Chennai to Hyderabad",
    ]
    st.write("Try one of these:")
    chip_cols = st.columns(len(example_queries))
    chip_clicked = None
    for col, ex in zip(chip_cols, example_queries):
        with col:
            if st.button(ex, key=f"chip_{ex}"):
                chip_clicked = ex

    # --- NEW: conversation history stored across reruns ---
    if "nl_history" not in st.session_state:
        st.session_state.nl_history = []

    query = st.text_input('Your question', value=chip_clicked or "", key="nl_query")

    if st.button('Search') and query:
        from groq import Groq

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        city_list = db.fetch_city_names()
        airline_list, _ = db.fetch_airline_frequency()

        # --- NEW: extraction now also covers airline preference and sort intent
        # ("cheapest" -> Price, "fastest" -> Duration) ---
        system_prompt = f"""
        Extract flight search parameters from the user's question.
        Valid cities: {city_list}
        Valid airlines: {airline_list}
        Respond ONLY with JSON, no other text: {{"source": "...", "destination": "...",
        "max_price": number or null, "non_stop_only": true/false,
        "airline": "..." or null, "sort_by": "Price" or "Duration"}}
        Use "Duration" for sort_by only if the user asks for fastest/quickest/shortest.
        Otherwise default sort_by to "Price". If a value isn't mentioned, use null
        (or false for non_stop_only).
        """

        with st.spinner('Parsing your request...'):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                max_tokens=200,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ]
            )

        import json
        try:
            parsed = json.loads(response.choices[0].message.content)
        except (json.JSONDecodeError, IndexError, AttributeError):
            st.error("Couldn't parse that request — try rephrasing with a clear source and destination.")
            parsed = None

        if parsed and parsed.get('source') and parsed.get('destination'):
            # --- NEW: show what was actually understood, so mistakes are visible ---
            filter_bits = [f"**{parsed['source']} → {parsed['destination']}**"]
            if parsed.get('airline'):
                filter_bits.append(f"airline: {parsed['airline']}")
            if parsed.get('max_price'):
                filter_bits.append(f"under ₹{parsed['max_price']:,.0f}")
            if parsed.get('non_stop_only'):
                filter_bits.append("non-stop only")
            filter_bits.append(f"sorted by {parsed.get('sort_by', 'Price').lower()}")
            st.caption("Detected: " + " · ".join(filter_bits))

            results = db.fetch_all_flights_filtered(
                parsed['source'], parsed['destination'],
                max_price=parsed.get('max_price'),
                non_stop_only=parsed.get('non_stop_only', False),
                sort_by=parsed.get('sort_by', 'Price'),
                airline=parsed.get('airline')
            )

            if results.empty:
                st.info('No flights match that description.')
                summary_text = None
            else:
                st.dataframe(results, use_container_width=True)

                # --- NEW: ask Groq for a one-line plain-English takeaway ---
                top = results.iloc[0]
                summary_prompt = (
                    f"In one short sentence, summarize this flight search result for a user. "
                    f"{len(results)} flights found. Cheapest/top option: {top['Airline']}, "
                    f"price ₹{top['Price']}, duration {top['Duration']}, stops {top['Total_Stops']}. "
                    f"Be conversational, no preamble."
                )
                summary_response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    max_tokens=80,
                    temperature=0.4,
                    messages=[{"role": "user", "content": summary_prompt}]
                )
                summary_text = summary_response.choices[0].message.content
                st.success(summary_text)

            st.session_state.nl_history.append({
                "query": query, "count": len(results), "summary": summary_text
            })

        elif parsed:
            st.warning("Couldn't identify both a source and destination city — try being more specific.")

    # --- NEW: show recent questions asked this session ---
    if st.session_state.nl_history:
        with st.expander("Previous questions this session"):
            for item in reversed(st.session_state.nl_history[-5:]):
                st.write(f"**Q:** {item['query']}")
                st.caption(f"{item['count']} results" + (f" — {item['summary']}" if item['summary'] else ""))

else:
    # --- NEW: designed landing / about page instead of plain text ---

    st.markdown("""
        <style>
        .hero {
            padding: 2.2rem 2rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
            border: 1px solid #2d3748;
            margin-bottom: 1.8rem;
        }
        .hero h1 {
            font-size: 2.1rem;
            margin-bottom: 0.4rem;
            color: #f9fafb;
        }
        .hero p {
            font-size: 1.05rem;
            color: #9ca3af;
            max-width: 640px;
            margin: 0;
        }
        .feature-card {
            background: #161b22;
            border: 1px solid #2d3748;
            border-radius: 12px;
            padding: 1.2rem 1.3rem;
            height: 100%;
        }
        .feature-card h4 {
            margin: 0 0 0.4rem 0;
            font-size: 1.02rem;
            color: #f3f4f6;
        }
        .feature-card p {
            margin: 0;
            font-size: 0.88rem;
            color: #9ca3af;
            line-height: 1.45;
        }
        .badge {
            display: inline-block;
            background: #1f2937;
            border: 1px solid #374151;
            border-radius: 999px;
            padding: 0.25rem 0.85rem;
            margin: 0.15rem 0.3rem 0.15rem 0;
            font-size: 0.82rem;
            color: #d1d5db;
        }
        .stat-box {
            text-align: center;
            padding: 0.8rem 0;
        }
        .stat-box .num {
            font-size: 1.7rem;
            font-weight: 700;
            color: #f9fafb;
        }
        .stat-box .label {
            font-size: 0.8rem;
            color: #9ca3af;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="hero">
            <h1>✈️ Flight Search & Analytics Platform</h1>
            <p>A full-stack data product: SQL-backed search, interactive analytics,
            a machine learning price predictor, and a natural-language interface
            powered by the Groq API — built end to end as a portfolio project.</p>
        </div>
    """, unsafe_allow_html=True)

    # --- live stats pulled from the actual database ---
    try:
        all_cities = db.fetch_city_names()
        airlines, freqs = db.fetch_airline_frequency()
        total_flights = sum(freqs)

        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(f'<div class="stat-box"><div class="num">{total_flights:,}</div>'
                        f'<div class="label">Flights indexed</div></div>', unsafe_allow_html=True)
        with s2:
            st.markdown(f'<div class="stat-box"><div class="num">{len(airlines)}</div>'
                        f'<div class="label">Airlines covered</div></div>', unsafe_allow_html=True)
        with s3:
            st.markdown(f'<div class="stat-box"><div class="num">{len(all_cities)}</div>'
                        f'<div class="label">Cities served</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
    except Exception:
        pass  # stats are a nice-to-have; don't break the page if a query fails

    # --- feature cards ---
    st.subheader("What's inside")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
            <div class="feature-card">
                <h4>🔍 Check Flights</h4>
                <p>Search by source/destination with filters for price cap, non-stop only,
                and sort order. Export any result set straight to CSV.</p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
            <div class="feature-card">
                <h4>🤖 Price Predictor</h4>
                <p>A RandomForest regression model trained on historical fares —
                predicts ticket price from airline, route, stops, and duration,
                with held-out error reported live.</p>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
            <div class="feature-card">
                <h4>📊 Analytics Dashboard</h4>
                <p>Flights per airline, busiest airports, daily volume, and price
                spread by airline — visualized with Plotly.</p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
            <div class="feature-card">
                <h4>💬 Ask in Plain English</h4>
                <p>Type a question like "cheapest non-stop flight from Delhi to Mumbai" —
                the Groq API extracts structured filters and runs the search for you.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- tech stack ---
    st.subheader("Built with")
    st.markdown("""
        <span class="badge">Streamlit</span>
        <span class="badge">MySQL</span>
        <span class="badge">pandas</span>
        <span class="badge">scikit-learn</span>
        <span class="badge">Plotly</span>
        <span class="badge">Groq API</span>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Use the menu on the left to explore each feature →")