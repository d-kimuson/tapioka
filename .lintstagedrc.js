module.exports = {
  "src/**/*.py": ["poetry run flake8", "poetry run black"],
  "**/*.{js,ts,json,md,toml,yaml}": ["prettier --write"],
};
