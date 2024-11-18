import React from 'react';
import { NavLink } from 'react-router-dom';

const BottomNav = () => {
  return (
    <div className="fixed bottom-0 w-full bg-gray-800 text-white flex justify-around py-3">
      <NavLink
        to="/"
        className={({ isActive }) =>
          `flex flex-col items-center ${
            isActive ? 'text-orange-500' : 'text-white'
          }`
        }
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="currentColor" viewBox="0 0 960 960">
          <path d="M240-200h120v-240h240v240h120v-360L480-740 240-560v360Zm-80 80v-480l320-240 320 240v480H520v-240h-80v240H160Zm320-350Z" />
        </svg>
        <span>Home</span>
      </NavLink>

      <NavLink
        to="/confusion-matrix"
        className={({ isActive }) =>
          `flex flex-col items-center ${
            isActive ? 'text-orange-500' : 'text-white'
          }`
        }
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="currentColor" viewBox="0 0 960 960">
          <path d="M280-280h160v-160H280v160Zm240 0h160v-160H520v160ZM280-520h160v-160H280v160Zm240 0h160v-160H520v160ZM200-120q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h560q33 0 56.5 23.5T840-760v560q0 33-23.5 56.5T760-120H200Zm0-80h560v-560H200v560Zm0-560v560-560Z" />
        </svg>
        <span>Confusion Matrix</span>
      </NavLink>
    </div>
  );
};

export default BottomNav;
