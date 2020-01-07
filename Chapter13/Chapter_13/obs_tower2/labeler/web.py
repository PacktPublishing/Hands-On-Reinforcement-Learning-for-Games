from flask import Flask, send_file, send_from_directory

app = Flask(__name__, static_url_path='')

@app.route('/')
def handle_root():
    return send_from_directory('.', 'index.html')

@app.route('/assets/<path:path>')
def handle_asset(path):
    return send_from_directory('assets', path)

if __name__ == '__main__':
    app.run()
