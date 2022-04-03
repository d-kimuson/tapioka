module.exports = {
  "src/**/*.py": [
    "poetry run flake8 --config ./pyproject.toml",
    "poetry run black",
  ],
  "**/*.{js,ts,json,md,toml,yaml}": ["prettier --write"],
};
