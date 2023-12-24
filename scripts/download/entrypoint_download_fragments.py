from scripts.download.download_fragments import batch_download_frags
from utility.fragments import FragmentHandler


def get_user_input(prompt, options):
    """Prompt the user for input and validate the response."""
    print(f"\n\033[97m{prompt}\033[0m")  # White color for prompt
    for index, option in enumerate(options, start=1):
        print(f"\033[97m{index}. {option}\033[0m")  # White color for options
    choice = input("\033[97mEnter your choice (number): \033[0m").strip()

    # Ensure the input is a valid number and within the options range
    while not (choice.isdigit() and 1 <= int(choice) <= len(options)):
        choice = input(
            "\033[91mInvalid input, please enter a valid number: \033[0m").strip()  # Red color for invalid input

    return int(choice)


def get_user_string_input(prompt):
    """Prompt the user for a string input and validate the response."""
    print(f"\n\033[97m{prompt}\033[0m")  # White color for prompt
    user_input = input("\033[97mEnter your input: \033[0m").strip()

    # Ensure the input is not empty
    while not user_input:
        user_input = input(
            "\033[91mInvalid input, please enter a non-empty string: \033[0m").strip()  # Red color for invalid input

    return user_input


def download_fragments():
    options = ["Yes", "No"]
    consider_labels_choice = get_user_input("Consider label files when downloading fragments?", options)
    consider_labels = (consider_labels_choice == 1)

    single_layer = False
    if consider_labels:
        label_options = ["1-layer labels", "4-layer labels"]
        label_choice = get_user_input("Choose label type:", label_options)
        single_layer = (label_choice == 1)

    fragment_list = FragmentHandler().get_inference_fragments()

    print(f"\n\033[92mDownload process initiated!\033[0m\n")
    batch_download_frags(fragment_list, consider_labels=consider_labels, single_layer=single_layer)


if __name__ == '__main__':
    download_fragments()
