import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from models import (Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator,
                    feature_loss, generator_loss, discriminator_loss)
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint

torch.backends.cudnn.benchmark = True

def train(rank, a, h):
    # Initialize distributed training if needed
    if h.num_gpus > 1:
        init_process_group(
            backend=h.dist_config['dist_backend'],
            init_method=h.dist_config['dist_url'],
            world_size=h.dist_config['world_size'] * h.num_gpus,
            rank=rank,
        )

    torch.manual_seed(h.seed)
    torch.cuda.manual_seed(h.seed)
    device = torch.device(f'cuda:{rank}')

    # Initialize models and move to device
    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    if rank == 0:
        print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print(f"Checkpoints directory: {a.checkpoint_path}")

    # Try loading checkpoints if available
    cp_g = scan_checkpoint(a.checkpoint_path, 'g_') if os.path.isdir(a.checkpoint_path) else None
    cp_do = scan_checkpoint(a.checkpoint_path, 'do_') if os.path.isdir(a.checkpoint_path) else None

    steps = 0
    last_epoch = -1
    if cp_g is not None and cp_do is not None:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank])
        mpd = DistributedDataParallel(mpd, device_ids=[rank])
        msd = DistributedDataParallel(msd, device_ids=[rank])

    # Set up optimizers and LR schedulers
    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(
        itertools.chain(msd.parameters(), mpd.parameters()),
        h.learning_rate, betas=[h.adam_b1, h.adam_b2]
    )

    if cp_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    # Prepare datasets and dataloaders
    training_filelist, validation_filelist = get_dataset_filelist(a)
    trainset = MelDataset(
        training_filelist, h.segment_size, h.n_fft, h.num_mels,
        h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax,
        n_cache_reuse=0, shuffle=(False if h.num_gpus > 1 else True),
        fmax_loss=h.fmax_for_loss, device=device, fine_tuning=a.fine_tuning,
        base_mels_path=a.input_mels_dir
    )
    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None
    train_loader = DataLoader(
        trainset, num_workers=h.num_workers, shuffle=False,
        sampler=train_sampler, batch_size=h.batch_size,
        pin_memory=True, drop_last=True
    )

    if rank == 0:
        validset = MelDataset(
            validation_filelist, h.segment_size, h.n_fft, h.num_mels,
            h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax,
            False, False, n_cache_reuse=0, fmax_loss=h.fmax_for_loss,
            device=device, fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir
        )
        validation_loader = DataLoader(
            validset, num_workers=1, shuffle=False, batch_size=1,
            pin_memory=True, drop_last=True
        )
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    # Optionally enable mixed precision (AMP) training
    use_amp = getattr(h, 'use_amp', False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    generator.train()
    mpd.train()
    msd.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print(f"Epoch: {epoch+1}")

        if h.num_gpus > 1 and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()

            # Move inputs to device
            x, y, _, y_mel = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).unsqueeze(1)
            y_mel = y_mel.to(device, non_blocking=True)

            # Forward pass through generator (using AMP if enabled)
            with torch.cuda.amp.autocast(enabled=use_amp):
                y_g_hat = generator(x)
                y_g_hat_mel = mel_spectrogram(
                    y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                    h.sampling_rate, h.hop_size, h.win_size,
                    h.fmin, h.fmax_for_loss
                )

            # --- Update Discriminators ---
            optim_d.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                # Multi-Period Discriminator (MPD)
                y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
                loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
                # Multi-Scale Discriminator (MSD)
                y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
                loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
                loss_disc_all = loss_disc_s + loss_disc_f

            scaler.scale(loss_disc_all).backward()
            scaler.step(optim_d)

            # --- Update Generator ---
            optim_g.zero_grad()
            # Adjust mel spectrograms so that the shapes match
            min_shape = min(y_mel.shape[-1], y_g_hat_mel.shape[-1])
            y_mel_trim = y_mel[..., :min_shape]
            y_g_hat_mel_trim = y_g_hat_mel[..., :min_shape]
            min_batch = min(y_mel_trim.shape[0], y_g_hat_mel_trim.shape[0])
            y_mel_trim = y_mel_trim[:min_batch]
            y_g_hat_mel_trim = y_g_hat_mel_trim[:min_batch]

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss_mel = F.l1_loss(y_mel_trim, y_g_hat_mel_trim) * 45

                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_f, _ = generator_loss(y_df_hat_g)
                loss_gen_s, _ = generator_loss(y_ds_hat_g)
                loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            scaler.scale(loss_gen_all).backward()
            scaler.step(optim_g)
            scaler.update()

            if rank == 0:
                # Logging to STDOUT
                if steps % a.stdout_interval == 0:
                    mel_error = F.l1_loss(y_mel_trim, y_g_hat_mel_trim).item()
                    print(f"Steps: {steps}, Gen Loss Total: {loss_gen_all.item():.3f}, "
                          f"Mel-Spec. Error: {mel_error:.3f}, s/b: {time.time() - start_b:.3f}")

                # Checkpointing at intervals
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    cp_path_g = os.path.join(a.checkpoint_path, f"g_{steps:08d}")
                    save_checkpoint(cp_path_g, {
                        'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()
                    })
                    cp_path_do = os.path.join(a.checkpoint_path, f"do_{steps:08d}")
                    save_checkpoint(cp_path_do, {
                        'mpd': (mpd.module if h.num_gpus > 1 else mpd).state_dict(),
                        'msd': (msd.module if h.num_gpus > 1 else msd).state_dict(),
                        'optim_g': optim_g.state_dict(),
                        'optim_d': optim_d.state_dict(),
                        'steps': steps, 'epoch': epoch
                    })

                # Tensorboard summaries
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all.item(), steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)

                # Validation
                if steps % a.validation_interval == 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0.0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x_val, y_val, _, y_mel_val = batch
                            x_val = x_val.to(device)
                            y_val = y_val.to(device)
                            y_mel_val = y_mel_val.to(device)
                            y_g_hat_val = generator(x_val)
                            y_g_hat_val_mel = mel_spectrogram(
                                y_g_hat_val.squeeze(1), h.n_fft, h.num_mels,
                                h.sampling_rate, h.hop_size, h.win_size,
                                h.fmin, h.fmax_for_loss
                            )
                            # Squeeze extra batch dimension if necessary
                            if y_g_hat_val_mel.dim() == 3 and y_g_hat_val_mel.shape[0] == 1:
                                y_g_hat_val_mel = y_g_hat_val_mel.squeeze(0)
                            min_time_steps = min(y_mel_val.shape[-1], y_g_hat_val_mel.shape[-1])
                            y_mel_val_trim = y_mel_val[..., :min_time_steps]
                            y_g_hat_val_mel_trim = y_g_hat_val_mel[..., :min_time_steps]
                            val_err_tot += F.l1_loss(y_mel_val_trim.squeeze(0), y_g_hat_val_mel_trim).item()

                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio(f'gt/y_{j}', y_val[0], steps, h.sampling_rate)
                                    sw.add_figure(f'gt/y_spec_{j}', plot_spectrogram(x_val[0].cpu().numpy()), steps)
                                sw.add_audio(f'generated/y_hat_{j}', y_g_hat_val[0], steps, h.sampling_rate)
                                y_hat_spec = mel_spectrogram(
                                    y_g_hat_val.squeeze(1), h.n_fft, h.num_mels,
                                    h.sampling_rate, h.hop_size, h.win_size,
                                    h.fmin, h.fmax
                                )
                                sw.add_figure(
                                    f'generated/y_hat_spec_{j}',
                                    plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()),
                                    steps
                                )

                        val_err = val_err_tot / (j + 1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)
                    generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print(f"Time taken for epoch {epoch + 1} is {int(time.time() - start)} sec\n")

    # ===== Final Checkpoint Saving and ONNX Export (only on rank 0) =====
    if rank == 0:
        # Save final checkpoints
        final_cp_path_g = os.path.join(a.checkpoint_path, "g_final")
        save_checkpoint(final_cp_path_g, {
            'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()
        })
        final_cp_path_do = os.path.join(a.checkpoint_path, "do_final")
        save_checkpoint(final_cp_path_do, {
            'mpd': (mpd.module if h.num_gpus > 1 else mpd).state_dict(),
            'msd': (msd.module if h.num_gpus > 1 else msd).state_dict(),
            'optim_g': optim_g.state_dict(),
            'optim_d': optim_d.state_dict(),
            'steps': steps,
            'epoch': epoch
        })
        print("Final model checkpoints saved.")

        # Prepare the generator for export
        export_model = generator.module if h.num_gpus > 1 else generator
        export_model.eval()

        # Save a backup file in case ONNX export fails
        backup_file = os.path.join(a.checkpoint_path, "generator_backup.pt")
        torch.save(export_model.state_dict(), backup_file)
        print(f"Backup generator file saved to {backup_file} for future re-exporting if necessary.")

        # Create a dummy input matching the expected shape.
        # For HiFi-GAN, the generator typically expects a mel-spectrogram of shape (batch_size, n_mels, time_steps)
        dummy_input = torch.randn(1, h.num_mels, 100, device=device)  # Adjust the time steps (here 100) as needed
        onnx_filename = os.path.join(a.checkpoint_path, "generator.onnx")
        
        # Attempt the ONNX export
        try:
            torch.onnx.export(
                export_model,
                dummy_input,
                onnx_filename,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['mel'],
                output_names=['audio'],
                dynamic_axes={
                    'mel': {0: 'batch_size', 2: 'time_steps'},
                    'audio': {0: 'batch_size', 1: 'time_steps'}
                }
            )
            print(f"ONNX model exported to {onnx_filename}")
        except Exception as e:
            print(f"ONNX export failed: {e}")
            print(f"You can re-export the model later using the backup file at {backup_file}")

def main():
    print("Initializing Training Process...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='./dataset/wavs')
    parser.add_argument('--input_mels_dir', default='./dataset/mels')
    parser.add_argument('--input_training_file', default='./dataset/train_files.txt')
    parser.add_argument('--input_validation_file', default='./dataset/validation_files.txt')
    parser.add_argument('--checkpoint_path', default='./cp_hifigan')
    parser.add_argument('--config', default='./config_v3.json')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)
    a = parser.parse_args()

    with open(a.config) as f:
        json_config = json.load(f)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print(f"Batch size per GPU: {h.batch_size}")
    else:
        h.num_gpus = 0

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h))
    else:
        train(0, a, h)

if __name__ == '__main__':
    main()
