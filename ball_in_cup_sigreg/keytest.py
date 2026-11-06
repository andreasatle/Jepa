"""
Standalone pynput diagnostic -- nothing to do with dm_control.

Run it, then press some keys (try arrows, and a couple letters). Press Esc
to quit.

    uv run python -m keytest

If NOTHING prints when you press keys: pynput isn't receiving events at
all -- almost certainly the macOS Accessibility permission issue. Fix:
System Settings -> Privacy & Security -> Accessibility -> enable the
terminal app you're running this from (Terminal.app, iTerm, etc. -- or
possibly the Python/uv binary itself, depending on how your terminal
launches it). Then fully QUIT and reopen that terminal app -- macOS often
won't pick up a freshly-granted permission for an already-running process,
just closing the window isn't enough.

If key presses DO print here: pynput itself is fine, and the problem is
something specific to how it's wired into the dm_control viewer instead --
a different bug, come back and we'll dig into that.
"""

from pynput import keyboard


def on_press(key):
    print("pressed:", key)


def on_release(key):
    print("released:", key)
    if key == keyboard.Key.esc:
        print("Esc released -- stopping.")
        return False


print("Listening for keys... press Esc to quit.")
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
    