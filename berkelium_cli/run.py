from textual.app import App
from textual.widgets import Static

class BerkeliumCLI(App):
    """A simple Textual app."""

    def compose(self):
        self.static = Static("Hello, World!", classes="message")
        yield self.static

    def on_mount(self):
        self.static.styles.background = "orange"


def main():
    """Run the BerkeliumCLI app."""
    app = BerkeliumCLI()
    app.run()


if __name__ == "__main__":
    main()
