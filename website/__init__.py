from flask import Flask, session
from flask_sqlalchemy import SQLAlchemy
from pymongo import MongoClient
import os
import random

db = SQLAlchemy()
DB_NAME = 'database.db'


def create_app():
    app = Flask(__name__)
    random.seed(0)

    # Secret key for sessions
    app.config['SECRET_KEY'] = os.urandom(24)

    # --- SQLite setup (optional) ---
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
    db.init_app(app)

    # --- MongoDB setup ---
    mongo_uri = "mongodb://localhost:27017"
    mongo_client = MongoClient(mongo_uri)
    app.mongo_db = mongo_client.healthapp

    # --- Blueprints ---
    from .views import views
    from .prediction import prediction
    from .messages import messages
    from .auth import auth
    from .chatbot import chatbot
    from .routes.disease_routes import disease_routes

    # --- Register Blueprints ---
    app.register_blueprint(disease_routes, url_prefix='/')
    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(prediction, url_prefix='/')
    app.register_blueprint(messages, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')
    app.register_blueprint(chatbot, url_prefix='/')

    # --- Make 'user' available in all templates ---
    @app.context_processor
    def inject_user():
        return dict(user=session.get('user'))

    # --- Prevent cached pages after logout ---
    @app.after_request
    def add_header(response):
        """
        Disable caching so that after logout, users can't go back to cached pages.
        """
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "-1"
        return response

    return app
