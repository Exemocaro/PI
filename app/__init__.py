from flask import Flask
from flask_socketio import SocketIO

socketio = SocketIO(async_mode="eventlet", max_http_buffer_size=10**7)

def create_app():
    app = Flask(__name__, template_folder='../templates', static_folder='../static')
    app.config['SECRET_KEY'] = 'your_secret_key'
    socketio.init_app(app)
    return app 
