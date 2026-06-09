from pathlib import Path

project_root = Path(__file__).resolve().parents[2]


def print_section_header(title: str, width: int = 70) -> None:
    """
    Prints a formatted section header to the console for pipeline visibility.

    Args:
        title: The text to display in the header.
        width: The total character width of the header.
    """
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width + "\n")
