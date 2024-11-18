import React, { useState } from 'react';

const cities = [
  'Agusan Del Norte', 'Albay', 'Aurora', 'Batanes', 'Batangas', 'Benguet', 'Bukidnon', 'Cagayan', 'Camarines',
  'Capiz', 'Catanduanes', 'Cavite', 'Cebu', 'Davao Del Sur', 'Eastern Samar', 'Ilocos Norte', 'Ilocos Sur',
  'Leyte', 'Maguindanao', 'Masbate', 'Metro Manila', 'Negros Oriental', 'Nueva Ecija', 'Northern Samar',
  'Occidental Mindoro', 'Palawan', 'Pampanga', 'Pangasinan', 'Quezon', 'Rizal', 'Romblon', 'Sorsogon',
  'South Cotabato', 'Southern Leyte', 'Surigao Del Norte', 'Surigao Del Sur', 'Western Samar', 'Zambales',
  'Zamboanga Del Norte'
];

const SearchBar = ({ onSelectCity }) => {
  const [query, setQuery] = useState('');
  const [suggestions, setSuggestions] = useState([]);

  const handleInputChange = (e) => {
    const value = e.target.value;
    setQuery(value);

    if (value) {
      const filteredSuggestions = cities.filter((city) =>
        city.toLowerCase().includes(value.toLowerCase())
      );
      setSuggestions(filteredSuggestions);
    } else {
      setSuggestions([]);
    }
  };

  const handleSuggestionClick = (city) => {
    setQuery(city);
    setSuggestions([]);
    onSelectCity(city);
  };

  const onSearch = () => {
    if (query) {
      const selectedCity = suggestions.find((city) => city.toLowerCase() === query.toLowerCase()) || query;
      handleSuggestionClick(selectedCity);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      onSearch();
    }
  };

  return (
    <div className="relative">
      <div className="relative w-full">
        <input
          type="text"
          value={query}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          placeholder="Enter province (Metro Manila, Cebu, ...)"
          className="w-full p-2 pr-10 border border-gray-300 rounded"
        />
        {/* Clickable Magnifying Glass Icon */}
        <svg
          onClick={onSearch}
          xmlns="http://www.w3.org/2000/svg"
          className="absolute right-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400 cursor-pointer"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-4.35-4.35m1.72-6.36a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
      </div>
      {suggestions.length > 0 && (
        <ul className="absolute w-full bg-white border border-gray-300 rounded mt-1">
          {suggestions.map((city) => (
            <li
              key={city}
              onClick={() => handleSuggestionClick(city)}
              className="p-2 cursor-pointer hover:bg-gray-200"
            >
              {city}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default SearchBar;
