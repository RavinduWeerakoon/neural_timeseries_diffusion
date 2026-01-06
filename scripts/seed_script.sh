PYTHONPATH=. python3 ntd/train_diffusion_model.py \
    base.experiment=seed_iv_exp \
    base.tag=conv_wn \
    dataset=seed_iv \
    diffusion=diffusion_linear_200 \
    diffusion_kernel=white_noise \
    network=ada_conv_seed_iv \
    network.signal_channel=12 \
    optimizer=base_optimizer \
    optimizer.num_epochs=1000 \
    optimizer.lr=0.0004 \
    +experiments/generate_samples=generate_samples \
    generate_samples.num_samples=1000

echo "DONE"

