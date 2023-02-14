import os, sys
sys.path.append(f"{os.path.abspath(os.pardir)}TTS/")
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig, BaseAudioConfig
from TTS.vocoder.configs import HifiganConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.gan import GAN

def train_glowtts(run_name:str, project_name:str, output_path:str, data_path:str, stats_path:str):
    dataset_config = BaseDatasetConfig(formatter="ljspeech", meta_file_train="metadata.csv", path=data_path)
    config = GlowTTSConfig(
        run_name=run_name,
        project_name=project_name,
        run_description=f"{run_name} User's {project_name} TTS (Glow-TTS)",
        save_checkpoints=False,
        batch_size=64,
        eval_batch_size=32,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        use_noise_augment=True,
        test_delay_epochs=0,
        epochs=1000,
        text_cleaner="korean_cleaners",
        use_phonemes=False,
        phoneme_language="ko-kr",
        print_step=25,
        print_eval=False,
        mixed_precision=False,
        compute_input_seq_cache=True,
        batch_group_size=4,
        loss_masking=True,
        grad_clip=0.05,
        lr=0.0001,
        output_path=output_path,
        datasets=[dataset_config],
        characters=CharactersConfig(
            pad="_",
            eos="~",
            bos="^",
            characters="\u1100\u1101\u1102\u1103\u1104\u1105\u1106\u1107\u1108\u1109\u110a\u110b\u110c\u110d\u110e\u110f\u1110\u1111\u1112\u1161\u1162\u1163\u1164\u1165\u1166\u1167\u1168\u1169\u116a\u116b\u116c\u116d\u116e\u116f\u1170\u1171\u1172\u1173\u1174\u1175\u11a8\u11a9\u11aa\u11ab\u11ac\u11ad\u11ae\u11af\u11b0\u11b1\u11b2\u11b3\u11b4\u11b5\u11b6\u11b7\u11b8\u11b9\u11ba\u11bb\u11bc\u11bd\u11be\u11bf\u11c0\u11c1\u11c2",
            punctuations= " .!?",
            phonemes="\u1100\u1101\u1102\u1103\u1104\u1105\u1106\u1107\u1108\u1109\u110a\u110b\u110c\u110d\u110e\u110f\u1110\u1111\u1112\u1161\u1162\u1163\u1164\u1165\u1166\u1167\u1168\u1169\u116a\u116b\u116c\u116d\u116e\u116f\u1170\u1171\u1172\u1173\u1174\u1175\u11a8\u11a9\u11aa\u11ab\u11ac\u11ad\u11ae\u11af\u11b0\u11b1\u11b2\u11b3\u11b4\u11b5\u11b6\u11b7\u11b8\u11b9\u11ba\u11bb\u11bc\u11bd\u11be\u11bf\u11c0\u11c1\u11c2",
            is_unique=True
        ),
        test_sentences=["모델 학습을 위해 사용하지 않은 문장들입니다."],
        audio=BaseAudioConfig(
            sample_rate=22050,
            preemphasis=0.98,
            do_trim_silence= False,
            trim_db=60,
            power=1.1,
            mel_fmax=8000.0,
            stats_path=stats_path
        )
    )
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    model = GlowTTS(config, ap, tokenizer, speaker_manager=None)
    trainer = Trainer(
        TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )
    trainer.fit()

def train_hifigan(run_name:str, project_name:str, output_path:str, data_path:str, stats_path:str):
    config = HifiganConfig(
        run_name=run_name,
        project_name=project_name,
        run_description=f"{run_name} User's {project_name} TTS (HiFi-GAN)",
        save_checkpoints=False,
        batch_size=32,
        eval_batch_size=16,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=0,
        epochs=1000,
        seq_len=8192,
        pad_short=2000,
        use_noise_augment=False,
        eval_split_size=10,
        print_step=25,
        print_eval=False,
        mixed_precision=False,
        lr_gen=0.0002,
        lr_disc=0.0002,
        data_path=data_path,
        use_cache=True,
        wd=0.0,
        feat_match_loss_weight=10.0,
        output_path=output_path,
        audio=BaseAudioConfig(
            preemphasis=0.98,
            do_trim_silence= False,
            trim_db=60,
            power=1.1,
            mel_fmax=8000.0,
            stats_path=stats_path
        )
    )
    ap = AudioProcessor.init_from_config(config)
    eval_samples, train_samples = load_wav_data(config.data_path, config.eval_split_size)
    model = GAN(config, ap)
    trainer = Trainer(
        TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )
    trainer.fit()