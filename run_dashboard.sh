#!/bin/bash
echo "Starting Dashboard..."
cd web_dashboard
python manage.py runserver 0.0.0.0:8000

