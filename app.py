"""numpycal: a single-process, loopback-only Flask GUI."""
from pathlib import Path
import threading
import webbrowser
import argparse
from flask import Flask, jsonify, request, render_template_string
from werkzeug.exceptions import HTTPException
import numpy as np
from drills import public_patterns, PATTERNS
from examples import public_examples
from engine import GROUPS, OPERATIONS, parse_array, packed, calculate, run_example
from shapes import predict_shape

app = Flask(__name__, static_folder=None)
app.config.update(MAX_CONTENT_LENGTH=256_000)


@app.after_request
def local_headers(response):
    response.headers['Content-Security-Policy'] = "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; img-src data:; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response


@app.get('/')
def index():
    return render_template_string(Path(__file__).with_name('ui.html').read_text(),
        groups=[dict(title=title,ops=[OPERATIONS[r[0]] for r in rows]) for title,rows in GROUPS],
        operations=OPERATIONS,patterns=public_patterns(),examples=public_examples())


@app.post('/api/parse')
def parse():
    p = payload()
    return jsonify(packed(parse_array(p.get('text'),p.get('mode','auto'))))


def payload():
    p=request.get_json()
    if not isinstance(p,dict): raise ValueError('Expected an object of calculator inputs.')
    return p


@app.post('/api/predict')
@app.post('/api/run')
def run():
    p=payload(); op=p.get('operation')
    if op not in OPERATIONS: raise ValueError('Unknown operation.')
    A=parse_array(p.get('A'),p.get('modeA','auto'))
    B=parse_array(p.get('B'),p.get('modeB','auto')) if OPERATIONS[op]['binary'] else None
    params=p.get('params',{})
    if not isinstance(params,dict): raise ValueError('Parameters must be an object.')
    if request.path == '/api/predict':
        return jsonify(predict_shape(op,A,B,params))
    return jsonify(calculate(op,A,B,params))


@app.post('/api/pattern')
def pattern():
    p=payload(); d=next((d for d in PATTERNS if d['id']==p.get('id')),None)
    if d is None: raise ValueError('Unknown pattern.')
    result=calculate(d['id'],np.array(d['A']),np.array(d['B']) if OPERATIONS[d['id']]['binary'] else None,d['params'])
    result.update(fixture=dict(A=d['A'],B=d['B']),params=d['params'],steps=d['steps'],note=d['note'])
    return jsonify(result)


@app.post('/api/example')
def example():
    return jsonify(run_example(payload().get('id')))


@app.errorhandler(Exception)
def error(exc):
    if isinstance(exc,HTTPException): return jsonify(error=exc.description),exc.code
    if isinstance(exc,(ValueError,TypeError,IndexError,KeyError,np.linalg.LinAlgError,FloatingPointError,OverflowError)):
        return jsonify(error=str(exc) or 'Check the array shapes and parameters.'),400
    app.logger.exception('Unexpected calculator error')
    return jsonify(error='Unexpected calculation error. See the local terminal.'),500


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Offline NumPy calculator')
    parser.add_argument('--no-browser',action='store_true',help='Do not open a browser automatically')
    args=parser.parse_args()
    # Bind before opening the browser so an occupied port fails cleanly.
    from werkzeug.serving import make_server
    server=make_server('127.0.0.1',8765,app,threaded=True)
    print('numpycal → http://localhost:8765   (Ctrl+C to stop)',flush=True)
    if not args.no_browser:
        threading.Timer(0.3,lambda:webbrowser.open('http://localhost:8765')).start()
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
