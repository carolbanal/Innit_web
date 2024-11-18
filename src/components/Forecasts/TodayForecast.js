import React from 'react';

const TodayForecast = ({ forecast, category, city, loading, error }) => {
  if (loading) {
    return <p className="text-center text-white mt-4">Loading...</p>;
  }

  if (error) {
    return <p className="text-red-500 text-center mt-4">{error}</p>;
  }

  if (forecast !== null) {
    return (
      <div className="pt-10">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-white">Today's Weather Forecast</h1>
          <p className="py-8 text-orange-500 text-lg font-bold">{category?.label}</p>
          <p className="pb-8 text-white text-8xl font-bold">{forecast}°C</p>
          <p className="pb-12 text-gray-300">Today's peak heat index would feel like</p>
        </div>

        {/* Center only "Today's Current Conditions" */}
        <div className="p-4 mt-20" style={{ backgroundColor: 'rgba(24, 32, 41, 1)' }}>
          <h1 className="text-white text-2xl text-center">Today's Current Conditions</h1>
          <div
            className="p-4 mt-4"
            style={{
              backgroundColor: 'rgba(249, 115, 22, 0.17)',
              borderWidth: 1,
              borderColor: 'rgba(249, 115, 22, 1)',
              borderStyle: 'solid',
              borderRadius: 12,
            }}
          >
            <p className="text-orange-500 font-bold">{category?.label}</p>
            <p className="text-white">{category?.description}</p>
          </div>
        </div>
      </div>
    );
  }

  return null;
};

export default TodayForecast;
