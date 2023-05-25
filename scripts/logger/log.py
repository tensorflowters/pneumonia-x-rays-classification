def title_important(message: str):
    print(
        "\n\033[91m"
        "=================================================================\n"
        f"{message}\n"
        "================================================================="
        "\033[0m"
    )


def title_success(message: str):
    print(
        "\n\033[92m"
        "=================================================================\n"
        f"{message}\n"
        "================================================================="
        "\033[0m"
    )


def title_warning(message: str):
    print(
        "\n\033[93m"
        "=================================================================\n"
        f"{message}\n"
        "================================================================="
        "\033[0m"
    )


def title_space(message: str):
    print(
        "\n\033[95m"
        "=================================================================\n"
        f"{message}\n"
        "================================================================="
        "\033[0m"
    )


def title_info(message: str):
    print(
        "\n\033[96m"
        "=================================================================\n"
        f"{message}\n"
        "================================================================="
        "\033[0m"
    )


def text_important(message: str):
    print(f"\n\033[91m{message}\033[0m")


def text_success(message: str):
    print(f"\n\033[92m{message}\033[0m")


def text_warning(message: str):
    print(f"\n\033[93m{message}\033[0m")


def text_space(message: str):
    print(f"\n\033[95m{message}\033[0m")


def text_info(message: str):
    print(f"\n\033[96m{message}\033[0m")
