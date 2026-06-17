"""干豆多分类项目统一入口。"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from prep.data_loader import load_processed_data, load_train_config
from prep.data_preprocess import run_preprocess
from src.feature_analysis import run_feature_analysis
from train_eval.evaluate import run_evaluate
from train_eval.train import run_train

PREPROCESS_CONFIG = PROJECT_ROOT / "config" / "preprocess.yaml"
TRAIN_CONFIG = PROJECT_ROOT / "config" / "train.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="干豆多分类机器学习项目")
    parser.add_argument(
        "--task",
        choices=["preprocess", "analyze", "train", "evaluate", "all"],
        default="preprocess",
        help="preprocess=预处理 | analyze=IQR/PCA | train=训练 | evaluate=测试评估 | all=全流程",
    )
    parser.add_argument("--config", default=str(PREPROCESS_CONFIG), help="预处理配置文件")
    parser.add_argument("--train-config", default=str(TRAIN_CONFIG), help="训练/测试配置文件")
    args = parser.parse_args()

    kw = {"config_path": args.config, "project_root": PROJECT_ROOT}

    if args.task in ("preprocess", "all"):
        run_preprocess(**kw)

    if args.task in ("analyze", "all"):
        run_feature_analysis(**kw)

    if args.task in ("train", "evaluate", "all"):
        train_config = load_train_config(PROJECT_ROOT, Path(args.train_config))
        data = load_processed_data(PROJECT_ROOT, train_config)

        results = None
        if args.task in ("train", "all"):
            results = run_train(data, train_config, PROJECT_ROOT)

        if args.task in ("evaluate", "all"):
            run_evaluate(data, train_config, PROJECT_ROOT, results=results)


if __name__ == "__main__":
    main()
