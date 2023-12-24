from get_inference import get_inference_folder
from utility.fragments import FragmentHandler
from utility.meta_data import AlphaBetaMeta


def get_inference_for_checkpoint(frag_ids, checkpoint_path, host):
    for frag_id in frag_ids:
        get_inference_folder(fragment_id=frag_id, full_model_run_dir=checkpoint_path, hostname=host, single=True)


if __name__ == '__main__':
    meta = AlphaBetaMeta()

    fragments = FragmentHandler().get_inference_fragments()
    checkpoint = meta.get_previous_model()

    get_inference_for_checkpoint(frag_ids=fragments, checkpoint_path=checkpoint, host="vast")
