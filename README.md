# ddpm-mnist


python main.py --log=0 --config="./configs/ddpm/cifar10.py" --workdir="./results/et_plus_cifar10/" --mode="train" --override_param="model.name=et_plus,training.n_epochs=50,training.batch_size=30,training.eval_freq=2,training.snapshot_freq=10"

