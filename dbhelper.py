# BACKEND FILE

import os
import mysql.connector
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # reads variables from a local .env file, if present


class DB:
    def __init__(self):
        self.connection_error = None
        try:
            self.conn = mysql.connector.connect(
                host=os.getenv("DB_HOST", "127.0.0.1"),
                user=os.getenv("DB_USER", "root"),
                password=os.getenv("DB_PASSWORD", ""),
                database=os.getenv("DB_NAME", "flights")
            )
            self.mycursor = self.conn.cursor()
            print('Connection established')
        except mysql.connector.Error as err:
            # NOTE: bare except: hides real errors (auth failure vs server down vs
            # wrong db name all look identical). Catching the specific exception
            # and printing it makes debugging + demoing this to interviewers easier.
            print(f'Connection error: {err}')
            self.conn = None
            self.mycursor = None
            self.connection_error = str(err)  # surfaced in the UI, see app.py

    # ---------- existing methods (unchanged) ----------

    def fetch_city_names(self):
        city = []
        self.mycursor.execute("""
        SELECT DISTINCT(Destination) FROM flights.flights
        UNION 
        SELECT DISTINCT(Source) FROM flights.flights 
        """)
        data = self.mycursor.fetchall()
        for item in data:
            city.append(item[0])
        return city

    def fetch_all_flights(self, source, destination):
        # NOTE: kept for backward compatibility, but prefer fetch_all_flights_filtered
        # below — this version is vulnerable to SQL injection via string formatting.
        self.mycursor.execute("""
        SELECT Airline,Route,Dep_Time,Duration,Price FROM flights.flights
        WHERE Source = %s AND Destination = %s
        """, (source, destination))
        return self.mycursor.fetchall()

    def fetch_airline_frequency(self):
        airline, frequency = [], []
        self.mycursor.execute("""
        SELECT Airline, COUNT(*) FROM flights.flights
        GROUP BY Airline 
        """)
        data = self.mycursor.fetchall()
        for item in data:
            airline.append(item[0])
            frequency.append(item[1])
        return airline, frequency

    def busy_airport(self):
        city, frequency = [], []
        self.mycursor.execute("""
        SELECT Source, Count(*) FROM (SELECT Source FROM flights.flights 
                                      UNION ALL 
                                      SELECT Destination FROM flights.flights) t
        GROUP BY t.Source 
        ORDER BY COUNT(*) DESC 
        """)
        data = self.mycursor.fetchall()
        for item in data:
            city.append(item[0])
            frequency.append(item[1])
        return city, frequency

    def daily_frequency(self):
        date, frequency = [], []
        self.mycursor.execute("""
        SELECT Date_of_journey, COUNT(*) FROM flights.flights
        GROUP BY Date_of_journey 
        """)
        data = self.mycursor.fetchall()
        for item in data:
            date.append(item[0])
            frequency.append(item[1])
        return date, frequency

    # ---------- NEW: filtered search (feature 1) ----------

    def fetch_all_flights_filtered(self, source, destination, max_price=None,
                                    non_stop_only=False, sort_by="Price", airline=None):
        """
        Parameterized query (safe from SQL injection, unlike the original
        .format() version) with optional price cap, non-stop filter, airline
        filter, and sort. sort_by must be one of a whitelist to avoid
        injecting into ORDER BY.
        """
        query = """
            SELECT Airline, Route, Dep_Time, Duration, Price, Total_Stops
            FROM flights.flights
            WHERE Source = %s AND Destination = %s
        """
        params = [source, destination]

        if max_price is not None:
            query += " AND Price <= %s"
            params.append(max_price)

        if non_stop_only:
            query += " AND Total_Stops = 0"

        if airline:
            query += " AND Airline = %s"
            params.append(airline)

        allowed_sorts = {"Price", "Duration"}
        sort_col = sort_by if sort_by in allowed_sorts else "Price"
        query += f" ORDER BY {sort_col} ASC"

        self.mycursor.execute(query, tuple(params))
        columns = [desc[0] for desc in self.mycursor.description]
        return pd.DataFrame(self.mycursor.fetchall(), columns=columns)

    # ---------- NEW: extra analytics (feature 5 support) ----------

    def fetch_price_by_airline(self):
        """Returns raw (airline, price) rows for a box plot of price spread per airline."""
        self.mycursor.execute("""
            SELECT Airline, Price FROM flights.flights
        """)
        data = self.mycursor.fetchall()
        return pd.DataFrame(data, columns=["Airline", "Price"])

    # ---------- NEW: training data for the price predictor (feature 2) ----------

    def fetch_training_data(self):
        """
        Pulls the columns needed to train a simple price predictor.
        Kept separate from analytics queries since ML feature needs are
        likely to grow (more columns, feature engineering) independent of
        what the dashboard displays.
        """
        self.mycursor.execute("""
            SELECT Airline, Source, Destination, Total_Stops, Duration, Price
            FROM flights.flights
        """)
        columns = [desc[0] for desc in self.mycursor.description]
        return pd.DataFrame(self.mycursor.fetchall(), columns=columns)