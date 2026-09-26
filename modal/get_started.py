"""Minimal Modal smoke test — run: modal run modal/get_started.py"""

import modal

app = modal.App("devin-meetup-rio-get-started")


@app.function()
def square(x: int) -> int:
    print("This code is running on a remote worker!")
    return x**2


@app.local_entrypoint()
def main() -> None:
    print("the square is", square.remote(42))
