from flask import Flask
app = Flask(__name__)
@app.route('/')
def hello_there():
    return 'Hello from MLOps Team 5!'
