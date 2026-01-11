/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./*.html",
    "./**/*.html",  // <--- The ** tells it to look in ALL subfolders
    "./**/*.js"
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        "primary": "#30e87a",
        "accent-blue": "#3b82f6",
        "background-light": "#f6f8f7",
        "background-dark": "#0f172a",
        "surface-dark": "#1e293b",
        "surface-border": "#334155",
      },
      fontFamily: {
        "display": ["Spline Sans", "sans-serif"],
        "mono": ["JetBrains Mono", "monospace"],
      },
      borderRadius: {
        "DEFAULT": "0.5rem",
        "lg": "1rem",
        "xl": "1.5rem",
        "2xl": "2rem",
        "full": "9999px"
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/container-queries'),
  ],
}