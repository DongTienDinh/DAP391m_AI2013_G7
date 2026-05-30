from pathlib import Path


project_root = Path(__file__).resolve().parents[2]


def print_section_header(title: str, width: int = 70) -> None:
    """
    Print a nice section header.
    """
    
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width + "\n")