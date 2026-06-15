⚡ FlareForecast
A niche health hackathon prototype: an AI-powered flare predictor for people with POTS / dysautonomia and similar chronic conditions.

Status: Planning / prototype stage. Not medical advice.

What it does
Tracks daily inputs: sleep, water, sodium, standing time, stress, meds, weather, and current symptoms.
Trains a small Random Forest on synthetic patient data.
Predicts the risk of a flare in the next 48 hours.
Explains the top risk drivers.
Shows a timeline of recent symptom history.
Why it wins hackathons
Niche & emotional: Targets a real, underserved patient community.
Clear demo: A dashboard with a risk gauge, live prediction, and timeline.
AI + data: ML model + explainability + visualization.
Feasible: Works end-to-end in a weekend.
Quick start
Backend
Bash

cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
API docs: http://localhost:8000/docs

Frontend
Open frontend/index.html in a browser, or serve it:

Bash

cd frontend
python -m http.server 3000
Then visit http://localhost:3000

Project structure
text

flareforecast/
├── backend/
│   ├── main.py            # FastAPI + ML model
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
└── README.md
Next steps for the hackathon
Replace synthetic data with a real wearable/healthkit integration.
Add user accounts and multi-day journaling.
Improve explainability (SHAP-style values).
Add a "what-if" simulator: "If I drink 500ml more water, how does risk change?"
Build a pitch deck around the patient story.
Team roles suggestion
Frontend: Polish the dashboard, animations, mobile responsiveness.
Backend/ML: Improve model, add more features, deploy to cloud.
DevOps: Docker + deploy backend + frontend.
Pitch/Design: User research, slide deck, demo script.
