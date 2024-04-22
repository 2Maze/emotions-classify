


def check_tuning_res(tune_dir):
    tune_root_dir = join(ROOT_DIR, "tmp", "tune", tune_dir)
    folders = [
        f for f in os.listdir(tune_root_dir)
        if os.path.isdir(join(tune_root_dir, f))
    ]
    results = []
    for one_tune_iteration_dir in folders:
        results_dir = join(ROOT_DIR, "tmp", "tune", tune_dir, one_tune_iteration_dir, "result.json")
        with open(results_dir, "r") as f:
            results.append(
                max([i for i in
                     (json.loads(i) for i in f.readlines())],
                    key=lambda x: x['val/acc'])
            )
    results.sort(key=lambda x: x['val/acc'], reverse=True)
    print(*[
        {
            "acc": i['val/acc'],
            "epoch": i['epoch'],
            "id": i['trial_id']
        } | i['config']['train_loop_config']
        for i in results], sep='\n')