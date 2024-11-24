from app import create_app, socketio
from app.routes import register_routes
from app.events import *

app = create_app()
register_routes(app)

if __name__ == '__main__':
    print("Starting Flask-SocketIO server...")
    socketio.run(app, host="0.0.0.0", port=5001)