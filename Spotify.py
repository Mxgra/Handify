import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json

with open('secrets.txt') as f:
    secrets = f.read()
secrets = json.loads(secrets)
#print(secrets)

scope = "user-library-read playlist-read-private playlist-read-collaborative user-modify-playback-state user-read-playback-state"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=secrets['client_id'],
                                               client_secret=secrets['client_secret'],
                                               redirect_uri='http://localhost:8888/callback',
                                               scope=scope))

results = sp.current_user_saved_tracks()
print("A total of {} songs was found".format(len(results['items'])))

for idx, item in enumerate(results['items']):

    track = item['track']
    print(idx, track['artists'][0]['name'], " – ", track['name'])
    if idx > 10:
        break

print("Devices with open spotify app for this user:")
print(sp.devices())


def play_playback(device_id=None):
    try:
        sp.start_playback(device_id=device_id)
    except spotipy.exceptions.SpotifyException as e:
        print(e)


def stop_playback(device_id=None):
    try:
        sp.pause_playback(device_id=device_id)
    except spotipy.exceptions.SpotifyException as e:
        print(e)



#sp.start_playback(device_id=secrets['device_id'])
res = sp.current_playback()
print(res)
#sp.pause_playback(device_id=secrets['device_id'])