# Handify
Small hobby projects to train and use a hand gesture model to start and stop spotify playback.

The model is trained to recognize one to five fingers lifted on each hand, as well as the rock'n'roll hand sign.
To use simply add a secrets.txt file with a json like so:

{"client_id":"abc",
"client_secret":"def",
"device_id":"ghi"}

Spotify must be running, then call Inference.py.
The rock'n'roll sign will start the current song, raising two fingers will stop it. Cancel the script like you would any other, ctrl+c
