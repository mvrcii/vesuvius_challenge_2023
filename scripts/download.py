from scripts.download.entrypoint_download_checkpoint import download_checkpoint
from scripts.download.entrypoint_download_fragments import get_user_input, download_fragments
from scripts.download.entrypoint_download_masks import download_masks


def main():
    options = ["Slices", "Masks", "Checkpoint", "Single Inference Result for Checkpoint", "All Inference Results for Checkpoint"]
    download_choice = get_user_input("What do you want to download?", options)

    if download_choice == 1:
        download_fragments()
    elif download_choice == 2:
        download_masks()
    elif download_choice == 3:
        download_checkpoint()
    # elif download_choice == 3:
    #     download_inference()
    # elif download_choice == 3:
    #     download_inference_for_checkpoint()


if __name__ == '__main__':
    main()
