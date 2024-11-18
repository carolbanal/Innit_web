import React, { useState } from 'react';
import SearchBar from '../components/Forecasts/Search/SearchBar';
import WeeklyForecast from '../components/Forecasts/WeeklyForecast';
import TodayForecast from '../components/Forecasts/TodayForecast';
import WhatCanYouDo from '../components/Forecasts/WhatCanDo';

const HomeScreen = () => {
  const [city, setCity] = useState('Metro Manila');
  const [todayForecast, setTodayForecast] = useState(null);
  const [weeklyForecast, setWeeklyForecast] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const BASE_URL = "https://innit.onrender.com"; // Update for production later.

  const fetchTodayForecast = async (selectedCity) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${BASE_URL}/forecast/today/${encodeURIComponent(selectedCity)}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const result = await response.json();
      setTodayForecast(result?.predicted_value ?? null);
    } catch (err) {
      console.error('Fetch error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  
  const fetchWeeklyForecast = async (selectedCity) => {
    setLoading(true);
    try {
      const response = await fetch(`${BASE_URL}/forecast/7days/${encodeURIComponent(selectedCity)}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const result = await response.json();
      setWeeklyForecast(Array.isArray(result) ? result : []);
    } catch (err) {
      console.error('Fetch error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  

  const handleCitySelect = (selectedCity) => {
    setCity(selectedCity);
    fetchTodayForecast(selectedCity);
    fetchWeeklyForecast(selectedCity);
  };

  const getCategory = (temperature) => {
    if (temperature >= 27 && temperature <= 32) {
      return {
        label: 'CAUTION',
        description: (
          <>
            <p>
              This level indicates mild heat exposure. While not immediately
              dangerous, prolonged exposure or physical exertion may lead to
              fatigue.
            </p>
            <p>
              It is essential to stay hydrated and take breaks in shaded or
              cooler areas to prevent exhaustion.
            </p>
          </>
        ),
      };
    } else if (temperature >= 33 && temperature <= 41) {
      return {
        label: 'EXTREME CAUTION',
        description: (
          <>
            <p>
              Heat poses a significant risk to health. Heat stroke, heat cramps,
              or heat exhaustion are possible outcomes with prolonged exposure or
              physical exertion.
            </p>
            <br />
            <p>
              Heat exhaustion symptoms may include:<br />
              Heavy sweating<br />
              Muscle cramps<br />
              Tiredness<br />
              Fainting<br />
              Nausea<br />
              Headache<br />
              Dizziness<br />
            </p>
            <br />
            <p>
              Heat stroke symptoms may include:<br />
              High body temperature<br />
              Fast, strong pulse<br />
              Confusion<br />
              Losing consciousness<br />
              Nausea<br />
              Headache<br />
              Dizziness<br />
            </p>
            <br />
            <p>
              Take extra precautions such as wearing lightweight and loose-fitting
              clothing, drinking plenty of fluids, and minimizing outdoor
              activities during peak heat hours.
            </p>
            <br />
            <p>
              Curious about the different classifications of heat index?
              <a
                href="https://www.cdc.gov/disasters/extremeheat/index.html"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-500 underline"
              >
                Learn more
              </a>
            </p>
          </>
        ),
      };
    } else if (temperature >= 42 && temperature <= 54) {
      return {
        label: 'DANGER',
        description: (
          <>
            <p>
              Heat cramps or heat exhaustion are likely outcomes. There's an
              increased possibility of heat stroke, which can be life-threatening.
            </p>
            <br />
            <p>
              Heat exhaustion symptoms may include:<br />
              Heavy sweating<br />
              Muscle cramps<br />
              Tiredness<br />
              Fainting<br />
              Nausea<br />
              Headache<br />
              Dizziness<br />
            </p>
            <br />
            <p>
              Heat stroke symptoms may include:<br />
              High body temperature<br />
              Fast, strong pulse<br />
              Confusion<br />
              Losing consciousness<br />
              Nausea<br />
              Headache<br />
              Dizziness<br />
            </p>
            <br />
            <p>
              It is crucial to stay indoors in air-conditioned spaces, if
              possible, and limit outdoor activities to early mornings or evenings
              when temperatures are lower.
            </p>
            <p>
              <strong>In case of Emergency, call the Hotlines:</strong>
              <br />
              National Emergency Hotline: 911
              <br />
              Philippine Red Cross: 143 | 8527-0000
            </p>
            <p>
              Curious about the different classifications of heat index?
              <a
                href="https://www.cdc.gov/disasters/extremeheat/index.html"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-500 underline"
              >
                Learn more
              </a>
            </p>
          </>
        ),
      };
    } else if (temperature >= 55) {
      return {
        label: 'EXTREME DANGER',
        description: (
          <>
            <p>
              Heat stroke is highly likely. Immediate action is necessary to
              prevent serious health consequences.
            </p>
            <br />
            <p>
              Heat exhaustion symptoms may include:<br />
              Heavy sweating<br />
              Muscle cramps<br />
              Tiredness<br />
              Fainting<br />
              Nausea<br />
              Headache<br />
              Dizziness<br />
            </p>
            <br />
            <p>
              Heat stroke symptoms may include:<br />
              High body temperature<br />
              Fast, strong pulse<br />
              Confusion<br />
              Losing consciousness<br />
              Nausea<br />
              Headache<br />
              Dizziness<br />
            </p>
            <br />
            <p>
              Stay indoors in air-conditioned environments, drink plenty of
              fluids, and avoid any outdoor activities. If you must be outside,
              take frequent breaks in the shade and wear light-colored clothing to
              reflect sunlight.
            </p>
            <p>
              <strong>In case of Emergency, call the Hotlines:</strong>
              <br />
              National Emergency Hotline: 911
              <br />
              Philippine Red Cross: 143 | 8527-0000
            </p>
            <p>
              Curious about the different classifications of heat index?
              <a
                href="https://www.cdc.gov/disasters/extremeheat/index.html"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-500 underline"
              >
                Learn more
              </a>
            </p>
          </>
        ),
      };
    }
    return null;
  };

  return (
      <div className="relative flex-1 bg-gray-800 p-0">
        {/* Background Image */}
        <img
          src="/assets/bg-map.png"
          alt="Background Map"
          className="absolute top-0 left-1/2 transform -translate-x-1/2 w-full sm:w-auto sm:max-w-[80vw] h-auto max-h-[80vh] object-cover z-0"
        />

        {/* Search Bar */}
        <div className="relative z-10 flex flex-col items-center justify-center py-10">
          <SearchBar onSelectCity={handleCitySelect} />

        {/* Today Forecast */} 
        <TodayForecast
          forecast={todayForecast}
          category={getCategory(todayForecast)}
          city={city}
          loading={loading}
          error={error}
        />

        <div className="pb-8" style={{ backgroundColor: 'rgba(24, 32, 41, 1)' }}>
          {/* What Can You Do section */}
          {todayForecast && (
            <div className="px-0 mx-0">
              <div className="p-4">
                <WhatCanYouDo todayForecast={todayForecast} />
              </div>
            </div>
          )}

          {/* Weekly Forecast */}
          {!loading && !error && weeklyForecast.length > 0 && (
            <div className="mt-6 text-center mb-8" >
              <h1 className="p-4 text-white text-2xl">7-Day Forecast for {city}</h1>
              <WeeklyForecast city={city} forecasts={weeklyForecast} />
            </div>
          )}
          </div>
        </div>
      </div>
  );
};

export default HomeScreen;
