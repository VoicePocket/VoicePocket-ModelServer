import os, sys
sys.path.append(f"{os.path.abspath(os.pardir)}TTS/")
import numpy as np
from tqdm import tqdm
from TTS.tts.datasets import load_tts_samples
from TTS.utils.audio import AudioProcessor
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.vocoder.configs.hifigan_config import HifiganConfig
from TTS.tts.configs.shared_configs import BaseAudioConfig, CharactersConfig, BaseDatasetConfig

def make_compute_statistics(data_path:str, out_path:str, model_name:str):
    dataset_config = BaseDatasetConfig(formatter="ljspeech", meta_file_train="metadata.csv", path=data_path)
    audioConfig = BaseAudioConfig(preemphasis=0.98, do_trim_silence= False, trim_db=60, power=1.1, mel_fmax=8000.0)
    if model_name == "Glow_TTS":
        CONFIG = GlowTTSConfig(
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
            audio=audioConfig
        )
    elif model_name =="HiFi_GAN":
        CONFIG = HifiganConfig(
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
            audio = audioConfig
        )
    else:
        print("Wrong Model Name")
        return None
    
    ap = AudioProcessor.init_from_config(CONFIG)
    dataset_items = load_tts_samples(CONFIG.datasets)[0]
    
    mel_sum = 0
    mel_square_sum = 0
    linear_sum = 0
    linear_square_sum = 0
    N = 0
    for item in tqdm(dataset_items):
        wav = ap.load_wav(item if isinstance(item, str) else item["audio_file"])
        linear = ap.spectrogram(wav)
        mel = ap.melspectrogram(wav)

        N += mel.shape[1]
        mel_sum += mel.sum(1)
        linear_sum += linear.sum(1)
        mel_square_sum += (mel**2).sum(axis=1)
        linear_square_sum += (linear**2).sum(axis=1)

    mel_mean = mel_sum / N
    mel_scale = np.sqrt(mel_square_sum / N - mel_mean**2)
    linear_mean = linear_sum / N
    linear_scale = np.sqrt(linear_square_sum / N - linear_mean**2)

    output_file_path = out_path
    stats = {}
    stats["mel_mean"] = mel_mean
    stats["mel_std"] = mel_scale
    stats["linear_mean"] = linear_mean
    stats["linear_std"] = linear_scale

    stats["audio_config"] = CONFIG.audio.to_dict()
    np.save(output_file_path, stats, allow_pickle=True)