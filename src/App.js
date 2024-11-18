import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import HomeScreen from './screens/HomeScreen';  // import your HomeScreen component
import ConfusionMatrix from './screens/ConfusionMatrix';  // import your ConfusionMatrix component


function App() {
  return (
    <Router>
      <div>
        <nav className="p-2 bg-gray-900 flex justify-center items-center space-x-8">
          <Link to="/" className="p-2 text-white cursor-pointer hover:bg-gray-700 px-4 py-2 rounded">
            Home
          </Link>
          <Link to="/confusion-matrix" className="p-2 text-white cursor-pointer hover:bg-gray-700 px-4 py-2 rounded">
            Confusion Matrix
          </Link>
        </nav>

        <Routes>
          <Route path="/" element={<HomeScreen />} />
          <Route path="/confusion-matrix" element={<ConfusionMatrix />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;