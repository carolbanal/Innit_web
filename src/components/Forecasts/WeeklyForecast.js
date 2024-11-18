import React from 'react';

const WeeklyForecast = ({ city, forecasts }) => {
    const today = new Date();
    const tomorrow = new Date(today);
    tomorrow.setDate(today.getDate() + 1);

    const nextSevenDays = Array.from({ length: 7 }, (_, i) => {
        const nextDate = new Date(tomorrow);
        nextDate.setDate(tomorrow.getDate() + i);
        return nextDate.toISOString().split('T')[0];
    });

    const filteredForecast = forecasts.filter(forecast => {
        const forecastDate = new Date(forecast.date).toISOString().split('T')[0];
        return nextSevenDays.includes(forecastDate);
    });

    return (
        <div className="w-full px-20">
            {filteredForecast.length > 0 ? (
                // Parent container with responsive layout
                <div className="flex flex-col sm:flex-row sm:space-x-4 sm:space-y-0 space-y-4">
                    {filteredForecast.map((item) => {
                        const date = new Date(item.date);
                        const formattedDate = date.toLocaleDateString('en-US', { weekday: 'long' });

                        return (
                            <div
                                key={item.date}
                                className="bg-white bg-opacity-25 p-4 rounded-lg flex sm:flex-col flex-row items-center justify-between sm:items-center w-full sm:w-24"
                            >
                                <p className="text-white text-sm font-bold py-2">{formattedDate}</p>
                                <svg
                                    xmlns="http://www.w3.org/2000/svg"
                                    width="32"
                                    height="32"
                                    viewBox="0 0 24 24"
                                    fill="rgba(251, 191, 36, 1)"
                                    className="mx-4"
                                >
                                    <path
                                        d="M6.993 12c0 2.761 2.246 5.007 5.007 5.007s5.007-2.246 5.007-5.007S14.761 
                                        6.993 12 6.993 6.993 9.239 6.993 12zM12 8.993c1.658 0 3.007 1.349 3.007 
                                        3.007S13.658 15.007 12 15.007 8.993 13.658 8.993 12 10.342 8.993 12 
                                        8.993zM10.998 19h2v3h-2zm0-17h2v3h-2zm-9 9h3v2h-3zm17 0h3v2h-3zM4.219 
                                        18.363l2.12-2.122 1.415 1.414-2.12 2.122zM16.24 6.344l2.122-2.122 1.414 1.414-2.122 
                                        2.122zM6.342 7.759 4.22 5.637l1.415-1.414 2.12 2.122zm13.434 10.605-1.414 1.414-2.122-2.122 1.414-1.414z"
                                        stroke="rgba(251, 191, 36, 1)"
                                    />
                                </svg>
                                <p className="text-white text-lg font-bold py-2">{item.predicted_value}°C</p>
                            </div>
                        );
                    })}
                </div>
            ) : (
                <p className="text-white text-2xl text-center">No weekly forecast data available.</p>
            )}
        </div>
    );
};

export default WeeklyForecast;
