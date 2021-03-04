import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from flask_mongoengine import MongoEngine

app = Flask(__name__)

app.config['MONGODB_SETTINGS'] = {
    'db': 'your_database',
    'host': 'localhost',
    'port': 27017
}
db = MongoEngine()
db.init_app(app)

#Prepare function to preprocess incoming json data
def prepare(incoming):
    global X
    accelero = incoming.json()['AcceleroReadings']

    """## Formatting Data"""
    for i in accelero:
        x,y,z = i['x'],i['y'],i['z']
        #print(x,"\n", y,"\n", z)

    xdat=pd.DataFrame.from_dict(x, orient = 'index')
    ydat=pd.DataFrame.from_dict(y, orient = 'index')
    zdat=pd.DataFrame.from_dict(z, orient = 'index')
    #print(xdat, ydat, zdat)

    xyz=pd.concat([xdat,ydat,zdat], ignore_index=False, axis=1)
    #print(xyz)

    xyz.reset_index(level=0, inplace=True)

    xyz.columns=["time","x","y","z"]
    xyz = xyz.apply(pd.to_numeric)
    #print(xyz)
    #xyz.info()
    #xyz.describe()

    """## Remove obvious outlier values"""

    xyz = xyz[(xyz['x'] <50) | (xyz['y']<50) | (xyz['z']<50)] #Remove all values that are above 50/ keep values below 50
    #xyz.describe()


    """## Model Frame Preparation"""

    data=xyz[["x","y","z"]]

    #print(data.isnull().sum())
    data = data.dropna()
    #print(data.isnull().sum())
  
    scaler1 = StandardScaler()
    X = scaler1.fit_transform(data)

    scaler2 = MaxAbsScaler()
    X = scaler2.fit_transform(X)
  
    scaled_X = pd.DataFrame(data = X, columns = ['x', 'y', 'z'])

    #Framing into colvolutions
    def get_frames(df, frame_size, hop_size):

        N_FEATURES = 3

        frames = []
        for i in range(0, len(df) - frame_size, hop_size):
            x = df['x'].values[i: i + frame_size]
            #print(x)
            y = df['y'].values[i: i + frame_size]
            z = df['z'].values[i: i + frame_size]
            frames.append([x, y, z])
  
        # Bring the segments into a better shape
        frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
  
        return frames

    Fs = 8
    frame_size = Fs*2
    hop_size = Fs*4
    X = get_frames(scaled_X, frame_size, hop_size)
    #print(X.shape)

    #reshaping
    a=X.shape
    a = a + (1,)
    X = X.reshape(a)
    #print(X.shape)
    
    return X

#Load model in a function so its not reloaded
def get_model():
    global model
    model = tf.keras.models.load_model("./fall_detect1.h5")
    print("Fall model Loaded!")

print("Loading Keras Model..")
get_model()

#Run the prediction model
@app.route('/predict',methods=['POST'])
def predict():
    '''
    Fetch Data from API, call prepare function to preprocess, get prediction in json
    '''
    response = request.get_json(force="True")
    #Feed response into prepare function
    prepare(response)

    prediction = model.predict_classes(X)

    #print(prediction)
    
    response = {
        'prediction': prediction[0]
        }
    
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
