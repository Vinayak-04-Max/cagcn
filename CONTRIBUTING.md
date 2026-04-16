# Contributing Guide

Thanks for contributing to the Wafer Dashboard.

## Development setup

1. Clone the repository and enter the project folder.
2. Create and activate a Python virtual environment.
3. Install Python dependencies with `pip install -r requirements.txt`.
4. Start the ML API with `uvicorn ml_api:app --reload --host 127.0.0.1 --port 8000`.
5. In a second terminal, install Node dependencies and run `npm start`.

## Code standards

- Keep backend logic modular and side-effect free where possible.
- Prefer descriptive naming over short abbreviations.
- Keep request/response contracts documented in [`README.md`](README.md).
- Validate runtime behavior before opening a PR.

## Pull request checklist

- [ ] `ml_api.py` starts successfully in a clean virtual environment.
- [ ] `npm start` serves the dashboard at `http://localhost:3000`.
- [ ] API endpoints respond as documented.
- [ ] README updates are included for behavior or setup changes.
