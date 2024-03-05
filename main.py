from flask import Flask, render_template,request
from flask_socketio import SocketIO
from moderationCode import *

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'
socketio = SocketIO(app)

@app.route('/')
def sessions():
    
    return render_template('session.html')

@app.route('/badwords')
def badwords():
    return render_template('index.html')

@app.route('/addBadWordForm',methods=['POST'])
def addBadWordForm():
    message=request.form['message']
    AddData(message)
    return render_template('index.html',err='Bad Message Added')

def messageReceived(methods=['GET', 'POST']):
    print('message was received!!!')
    
@socketio.on('my event')
def handle_my_custom_event(json, methods=['GET', 'POST']):
    print('received my event: ' + str(json))
    print(json['message'])
    try:
        status=passMessage(json['message'])
        if status==True:
            socketio.emit('my response', json, callback=messageReceived)
    except:
        pass

if __name__ == '__main__':
    socketio.run(app, debug=True,port=5001)