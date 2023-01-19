import argparse
import logging
import multiprocessing
import os
import sys
import importlib.util

import rlsrl.api.config
from rlsrl.legacy import algorithm, environment, experiments
import rlsrl.base.name_resolve
import rlsrl.system

rlsrl.base.name_resolve.reconfigure("memory")
LOG_FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
DATE_FORMAT = "%Y%m%d-%H:%M:%S"

logger = logging.getLogger("SRL")


def run_worker(worker_type, config):
    workers = {
        "actor": rlsrl.system.basic.actor_worker.ActorWorker,
        "policy": rlsrl.system.basic.policy_worker.PolicyWorker,
        "trainer": rlsrl.system.basic.trainer_worker.TrainerWorker,
        "eval_manager": rlsrl.system.basic.eval_manager.EvalManager,
    }
    worker_class = workers[worker_type]
    worker = worker_class()
    worker.configure(config=config)
    worker.start()
    worker.run()

def config_local_worker_and_run(worker_type, worker_configs):
    logger.info(f"Running {len(worker_configs)} {worker_type} worker(s).")
    ps = [
        multiprocessing.Process(target=run_worker, args=(worker_type, c, ))
        for _, c in enumerate(worker_configs)
    ]
    return ps

def import_file(file_path: str):
    module_name = os.path.basename(file_path).strip(".py")
    assert os.path.exists(file_path), "Imported file does not exist."
    assert file_path.endswith(".py"), "Imported file is not a `.py` file."
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError:
        sys.path.append(os.path.dirname(file_path))
        spec.loader.exec_module(module)
    logger.info(f"Imported file: {file_path}")
    

def import_files(files):
    if not files:
        return 
    files = files.split(";")
    for f in files:
        import_file(f)


def run_local(args):
    import_files(args.import_files)

    exps = rlsrl.api.config.make_experiment(args.experiment_name)
    if len(exps) > 1:
        raise NotImplementedError()
    exp: rlsrl.api.config.Experiment = exps[0]
    setup = exp.initial_setup()
    setup.set_worker_information(args.experiment_name, args.trial_name)
    logger.info(
        f"Running {exp.__class__.__name__} experiment_name: {args.experiment_name} trial_name {args.trial_name}"
    )

    sample_streams = {}

    for e in setup.eval_managers:
        if isinstance(e.eval_sample_stream, str):
            if e.eval_sample_stream not in sample_streams.keys():
                q = multiprocessing.Queue()
                sample_streams[e.eval_sample_stream] = rlsrl.system.basic.local_sample.make_local_pair(q)
            e.eval_sample_stream = sample_streams[e.eval_sample_stream][1]
        else:
            raise NotImplementedError()

    for t in setup.trainer_workers:
        if isinstance(t.sample_stream, str):
            if t.sample_stream not in sample_streams.keys():
                q = multiprocessing.Queue()
                sample_streams[t.sample_stream] = rlsrl.system.basic.local_sample.make_local_pair(q)
            t.sample_stream = sample_streams[t.sample_stream][1]
        else:
            raise NotImplementedError()

    inference_streams = {}
    for a in setup.actor_workers:
        for i, inf in enumerate(a.inference_streams):
            if isinstance(inf, str):
                req_q = multiprocessing.Queue()
                resp_q = multiprocessing.Queue()
                a.inference_streams[i] = rlsrl.system.basic.local_inference.make_local_client(req_q, resp_q)
                if inf not in inference_streams.keys():
                    inference_streams[inf] = [(req_q, resp_q)]
                else:
                    inference_streams[inf].append((req_q, resp_q))
        for i, spl in enumerate(a.sample_streams):
            if isinstance(spl, str):
                a.sample_streams[i] = sample_streams[spl][0]

    for i, p in enumerate(setup.policy_workers):
        if isinstance(p.inference_stream, str):
            if p.inference_stream not in inference_streams.keys():
                raise KeyError(p.inference_stream)
            p.inference_stream = rlsrl.system.basic.local_inference.make_local_server(
                req_qs=[q[0] for q in inference_streams[p.inference_stream][i::len(setup.policy_workers)]],
                resp_qs=[q[1] for q in inference_streams[p.inference_stream][i::len(setup.policy_workers)]])
        else:
            raise NotImplementedError()

    workers = config_local_worker_and_run("actor", setup.actor_workers) + \
        config_local_worker_and_run("policy", setup.policy_workers) + \
        config_local_worker_and_run("trainer", setup.trainer_workers)

    for w in workers:
        w.start()

    for w in workers:
        w.join(timeout=3000)


def main():
    parser = argparse.ArgumentParser(prog="srl-local")
    subparsers = parser.add_subparsers(dest="cmd", help="sub-command help")
    subparsers.required = True

    subparser = subparsers.add_parser("run", help="starts a basic experiment")
    subparser.add_argument("--experiment_name", "-e", type=str, required=True, help="name of the experiment")
    subparser.add_argument("--trial_name", "-f", type=str, required=True, help="name of the trial")
    subparser.add_argument("--wandb_mode",
                           type=str,
                           default="disabled",
                           choices=["online", "offline", "disabled"])
    subparser.add_argument("--import_files", type=str, required=False, default="", help="Files to import, path split by `;`.")

    subparser.add_argument("--LOGLEVEL", type=str, default="INFO")
    subparser.set_defaults(func=run_local)

    args = parser.parse_args()
    logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT, level=getattr(args, "LOGLEVEL", "INFO"))
    args.func(args)


if __name__ == '__main__':
    main()
