# FightCast.app 
> Live Production (V1): https://fightcast.app  
> Infrastructure: Docker + Streamlit + Random Forest + Railway

![FightCast Demo](https://github.com/lbransby1/lbransby1/blob/main/MMAMetrics.gif?raw=true)

## Project Overview
FightCast is an end-to-end analytical platform designed to quantify stylistic mismatches in Mixed Martial Arts (MMA). The system moves beyond simple win/loss records by capturing non-linear fighter dynamics and stylistic "leaks" (e.g., Elite Grappling vs. Takedown Defense).

### Current Deployment (Baseline)
To ensure high system reliability and low inference latency for the initial launch, the production environment utilizes a Random Forest Classifier.
* Why Random Forest? Selected for its robust handling of outliers and non-parametric nature, providing a stable "Production Baseline" to verify the Dockerized deployment pipeline.
* Feature Engineering: Integrated domain-specific features including Reach-to-Height ratios, Age deltas, and stylistic fighter archetypes.

---

## Future Work & Roadmap

#### 🚀 Model Supercharging: The Ensemble Meta-Learner
I am currently bench-testing a high-performance Ensemble Meta-Learner to maximize predictive accuracy.
* The Tech: Stacking XGBoost, LightGBM, and Logistic Regression to capture subtle stylistic interactions that a single model might miss.
* Status: Research phase. View the experimental architecture and performance benchmarks here: [Link to your Ensemble Repo].

#### 🧠 Explainable AI (XAI): SHAP & LLM Integration
To transition from "Black Box" predictions to actionable insights:
* Advanced Features: Implementing temporal pre-processing to provide more nuanced SHAP (SHapley Additive exPlanations) values at inference time.
* Natural Language Explanations: Feeding high-impact SHAP values into an LLM (Large Language Model) to output human-readable fight breakdowns and prediction justifications.

#### 🛠️ System Robustness: Unit Testing & CI/CD
* Implementing a rigorous Unit Testing suite to validate model inputs/outputs and ensure edge-case fighters (e.g., debutants) don't break the inference pipeline.
* Automating performance regression tests to ensure new model iterations outperform the baseline before deployment.

---

## Technical Implementation
* Inference Engine: Scikit-learn based pipeline optimized for low-latency response times.
* Bias Correction: Implemented "Corner-Agnostic" training—swapping fighter positions during inference to ensure the model remains invariant to corner placement (Red vs. Blue).
* Engineering: Fully containerized with Docker to ensure absolute environment parity between development and Railway production.

## Local Development
```bash
# Clone the repository
git clone [https://github.com/lbransby1/MMAMetrics.git](https://github.com/lbransby1/MMAMetrics.git)

# Build the production-ready Docker image
docker build -t fightcast-v1 .

# Launch the baseline dashboard
docker run -p 8501:8501 fightcast-v1
