import streamlit as st
import pickle
import pandas as pd
import gdown
import os

teams = ['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# ---> NEW CODE TO DOWNLOAD MODEL FROM GOOGLE DRIVE <---
if not os.path.exists('pipe.pkl'):
    # Download the model from Google Drive
    url = 'https://drive.google.com/uc?id=1gf2N80tKNErfXIX6MgHSaKWNn5Rjo000'
    gdown.download(url, 'pipe.pkl', quiet=False)

# Load the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
# ---> END OF NEW CODE <---

st.title('IPL Win Predictor')

# Modern Streamlit columns
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

# Added type casting (int) to fix PyCharm warnings
target = int(st.number_input('Target', min_value=1, value=180))

col3, col4, col5 = st.columns(3)

with col3:
    score = int(st.number_input('Score', min_value=0, value=100))
with col4:
    # Type cast to float to handle overs like 10.5
    overs = float(st.number_input('Overs completed', min_value=0.1, max_value=20.0, value=10.0, step=0.1))
with col5:
    wickets_out = int(st.number_input('Wickets out', min_value=0, max_value=10, value=2))

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - int(overs * 6)

    # Safety checks to prevent crashes
    if runs_left <= 0:
        st.success("Batting team has already won! 🎉")
    elif balls_left <= 0:
        st.error("Match is over! No balls left.")
    else:
        # Model expects wickets REMAINING, so we subtract from 10
        wickets_remaining = 10 - wickets_out

        # Safe division (prevents crashes if overs is 0)
        crr = score / overs if overs > 0 else 0.0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0.0

        # Create dataframe for the model
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_remaining],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Make prediction
        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        # Display results
        st.header(batting_team + " - " + str(round(win * 100)) + "%")
        st.header(bowling_team + " - " + str(round(loss * 100)) + "%")