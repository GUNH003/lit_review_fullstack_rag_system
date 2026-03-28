/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx,css}'
  ],
  theme: {
    extend: {
      colors: {
        background: '#09090b', 
        surface: '#18181b',    
        primary: '#2563eb',
      }
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}