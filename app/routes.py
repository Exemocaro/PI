from flask import render_template
from app import create_app

def register_routes(app):
    @app.route('/')
    def index():
        return render_template('index.html')