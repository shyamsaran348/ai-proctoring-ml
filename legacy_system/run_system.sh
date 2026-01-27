#!/bin/bash
echo "--- Setting up AI Proctoring System (ML Integrated) ---"

# 1. Install Dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# 2. Apply Migrations
echo "Applying database migrations..."
python3 manage.py migrate

# 3. Create Default Superuser (if not exists - this is hacky in bash, skipping strictly automatic creation to avoid errors, user can do it if needed)
# But we can try to ensure the DB is usable.

# 4. Start Server
echo "Starting Django Server..."
echo "Access the site at http://127.0.0.1:8000"
python3 manage.py runserver 0.0.0.0:8000
