from textual.app import App
from textual.widgets import Static
from textual_pyfiglet import FigletWidget


class BerkeliumCLI(App):
    """Berkelium CLI app"""

    TITLE = "Berkelium CLI"
    SUB_TITLE = "Context Engineering Tool"

    CSS_PATH = "styles.tcss"

    def compose(self):
        yield FigletWidget(
            "Berkelium CLI",
            font="ansi_shadow",
            justify="center",
            colors=["#e6a08f", "#e05d38"],
            animate=True,
            classes="bkc-title",
        )
        self.static = Static("Context Engineering Tool", classes="bkc-subtitle")
        yield self.static

    def on_mount(self):
        self.static.styles.background = "#e6a08f"


def main():
    """Run the Berkelium CLI app."""
    app = BerkeliumCLI()
    app.run()


if __name__ == "__main__":
    main()
