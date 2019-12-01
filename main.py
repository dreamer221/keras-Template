from data_loader.data_generator import DataGenerator
from models.simple_conv_model import SimpleConvModel
from trainers.simple_conv_model_trainer import SimpleConvModelTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args


def main():
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("配置参数还没有指定")
        exit(0)

    create_dirs([config.summary_dir, config.checkpoint_dir])

    print('准备数据')
    data_generator = DataGenerator(config)

    print('准备模型')
    model = SimpleConvModel(config, data_generator.get_word_index())

    print('准备训练')
    trainer = SimpleConvModelTrainer(model.model, data_generator.get_train_data(), config)

    print('开始训练')
    trainer.train()

    print('可视化展示')
    trainer.visualize()


if __name__ == '__main__':
    main()
    # 使用添加参数的方式运行 -c configs/simple_conv_model.json
