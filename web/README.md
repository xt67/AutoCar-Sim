# Browser research dashboard

Deploy on Vercel with Root Directory `web` and framework Other. No build command or secrets are needed.
Serve locally with `python -m http.server 8080 --directory web`.

Data copied unchanged from `logs/baseline/history.json`, `logs/augmented/history.json`
and `docs/validation/carla-20261111.json` on 18 November 2026. Refresh these copies
after updating experiments. The GIF and video link point to existing repository media.

This is an interactive results dashboard, not a browser port of CARLA or a trained
driving model. The local simulator, training pipeline and Streamlit dashboard remain
unchanged. Legacy offline split provenance is unverified. Collision counts represent
flagged frames, not separate collision events.
