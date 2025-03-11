#USE %matplotlib widget

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import FloatSlider, Layout, Output, VBox, Button, HBox
from pyannote.core import notebook, Segment
from IPython.display import display, clear_output
from just_playback import Playback
import asyncio
import tempfile
import soundfile as sf

def player(audio, annotation, sample_rate=16000):

    # If `audio` is a path (string), load it
    if isinstance(audio, str):
        audio, sample_rate = sf.read(audio)

    duration = len(audio) / sample_rate

    plt.ioff() #avoid dupplicated plots

    # Initialize variables
    duration = len(audio) / sample_rate

    # Save 'c1' to a temporary WAV file
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    sf.write(temp_wav.name, audio, sample_rate)

    # Initialize the Playback object
    playback = Playback()
    playback.load_file(temp_wav.name)

    # Initialize plot elements
    fig, ax = plt.subplots(figsize=(10, 4))
    # Configure the notebook for visualization
    notebook.width = 100
    notebook.crop = Segment(0, duration)
    notebook.plot_annotation(annotation, ax=ax, legend=True, time=True)
  
    # Add a vertical line to indicate current time
    current_time_line = ax.axvline(x=0, color='black', linestyle='--', label='Current Time')

    # Create a slider widget to control the current time
    time_slider = FloatSlider(
        value=0,
        min=0,
        max=duration,
        step=0.1,
        description='Time (s):',
        continuous_update=True,
        layout=Layout(width='800px')
    )

    # Function to handle slider change events
    def on_time_change(change):
        current_time = change['new']
        # Seek playback only if the change was from the slider, not from playback progress
        if abs(current_time - playback.curr_pos) > 0.1:  # Avoid feedback loop
            playback.seek(current_time)
        # Update the vertical line position
        current_time_line.set_xdata([current_time, current_time])
        ax.figure.canvas.draw_idle()

    # Observe slider for changes
    time_slider.observe(on_time_change, names='value')

    # Asynchronous function to update the slider based on playback position
    async def update_slider():
        while True:
            if playback.playing:
                new_pos = playback.curr_pos
                if abs(new_pos - time_slider.value) > 0.1:
                    time_slider.value = new_pos
            await asyncio.sleep(0.1)  # Update every 100ms

    # Create Play, Pause, and Stop buttons
    play_button = Button(description="Play", button_style='success')
    pause_button = Button(description="Pause", button_style='warning')
    stop_button = Button(description="Stop", button_style='danger')

    # Define what happens when Play, Pause, and Stop buttons are clicked
    def on_play_button_clicked(b):
        if playback.active:
            playback.resume()
        else:
            playback.play()

    def on_pause_button_clicked(b):
        playback.pause()

    def on_stop_button_clicked(b):
        playback.stop()
        time_slider.value = 0  # Reset the slider to the beginning
        # Reset the vertical line position
        current_time_line.set_xdata([0, 0])
        ax.figure.canvas.draw_idle()

    # Bind the button click events
    play_button.on_click(on_play_button_clicked)
    pause_button.on_click(on_pause_button_clicked)
    stop_button.on_click(on_stop_button_clicked)

    # Arrange buttons in a horizontal box
    buttons = HBox([play_button, pause_button, stop_button])

    # Start the asynchronous update loop
    loop = asyncio.get_event_loop()
    loop.create_task(update_slider())

    # Display the controls and output plot
    display(VBox([buttons, time_slider, fig.canvas]))
    fig.canvas.header_visible = False

def __main__():
    player()