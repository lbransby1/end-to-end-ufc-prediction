## UFC Fight Predictor Dashboard

A production-grade Machine Learning application that compares fighter statistics and predicts outcomes using an ensemble of models.

### Architecture
- **Frontend:** Streamlit dashboard for data visualization and user interaction.
- **Backend:** FastAPI high-performance asynchronous API for model inference.
- **Containerization:** Fully Dockerized architecture managed via Docker Compose.
- **ML Logic:** Features swap-averaging inference to eliminate positional (Red/Blue corner) bias.

### Tech Stack
- **Languages:** Python (Pandas, NumPy, Scikit-Learn)
- **Deployment:** Docker, FastAPI, Streamlit, Railway
- **Data:** Scraped UFC athlete data with custom feature engineering (Elo ratings, strike accuracy, etc.)
