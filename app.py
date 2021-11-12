import streamlit as st
import json
import spotipy
from spotify_client import *
import torch
from model import Model, get_emb

model = Model()
model_path = 'best_model.pth'
model.load_state_dict(torch.load(model_path, map_location = torch.device('cpu')))

model.eval()

def predict_hit(lyrics,features):
    # lyrics
    emb = get_emb(lyrics)
    emb = torch.tensor(emb).float() 
    features = torch.tensor(features).float().reshape(1,15)


    with torch.no_grad():
        score = torch.nn.Sigmoid()(model(emb,features))

    return score *100

def main():
    # st.title("Hit Song Prediction")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Hit Song Prediction </h2>
    </div>
    """
    safe_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> Yayy!! </h2>
       </div>
    """
    danger_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> :( </h2>
       </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    song_name = st.text_input("Search song","Levitating")
    #search for artist dynamically
    search_results = spotify.search({"track":song_name})
    ids = []
    artis = []
    for item in search_results['tracks']['items']:
      ids.append(item['id'])
      artists = []
      for artist in item['artists']:
        artists.append(artist['name'])
      artis.append(','.join(artists))
    search_results = {artists : id for id,artists in zip(ids,artis)}
    if len(artis) == 0:
      st.write("No such song is found.. try again")
    else:
      sblist = list(set(artis))
      artist = st.selectbox("Select artist",sblist)
      song_features = (spotify.features(search_results[artist]),search_results[artist])
      keys = song_features[0].keys()
      features = [song_features[0][key] for key in keys if key in ['danceability','energy', 'liveness', 'tempo', 'explicit']]
      genre_onehot = [0]*10
      # genre_onehot[song_features[0]['genre']] = 1
      genre_onehot[6]=1
      features = features +[0]+ genre_onehot
      song_lyrics = get_lyrics(song_name,artist.split(',')[0])
      song_lyrics    

    if st.button("Predict"):
        output=predict_hit(song_lyrics, features)
        st.success('Song name is {} by {}'.format(song_name,artist))
        output
        if output < 0.8:
            st.markdown(danger_html,unsafe_allow_html=True)
        else:
            st.markdown(safe_html,unsafe_allow_html=True)

if __name__=='__main__':
    main()
