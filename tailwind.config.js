/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      scrollbar: ['rounded'], // Enables rounded scrollbars
    },
  },
  plugins: [
    require('tailwind-scrollbar'),
  ],
};
