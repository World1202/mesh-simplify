from .ui.app import main as ui_main


def run(argv: list[str] | None = None) -> int:
    return ui_main(argv)
