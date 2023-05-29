import os, sys
sys.path.append(f"{os.path.dirname(os.path.abspath(os.path.dirname(__file__)))}/")
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.bin.resample import resample_files
from pathlib import Path
from zipfile import ZipFile
import shutil

def build_dataset(file_path:str):
    data_path = Path(file_path).absolute().parent.__str__()
    # Build Dataset
    os.mkdir(f"{data_path}/wavs")
    shutil.move(file_path, f"{data_path}/wavs/temp.zip")
    unzip_audioFile(f"{data_path}/wavs/temp.zip")
    for file in os.listdir(f"./resources/"):
        shutil.copy(f"./resources/{file}", f"{data_path}/{file}")
    # Resample (NEED!)
    resample_files(data_path, 22050)    
    return data_path
    
def unzip_audioFile(path:str):
    path:Path = Path(path)
    with ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(path.parent)

def train_vits(run_name:str, project_name:str, output_path:str, data_path:str) -> Trainer:
    dataset_config = BaseDatasetConfig(formatter="sleeping_ce", meta_file_train="metadata.csv", path=data_path)
    
    audio_config = VitsAudioConfig(
        sample_rate=22050,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        mel_fmin=0,
        mel_fmax=8000,
    )
    config = VitsConfig(
        audio = audio_config,
        run_name=run_name,
        project_name=project_name,
        run_description=f"{run_name} User's {project_name} TTS",
        save_checkpoints=False,
        batch_size=16,
        eval_batch_size=32,
        batch_group_size=5,
        num_loader_workers=0,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        text_cleaner="korean_cleaners",
        use_phonemes=False,
        phoneme_language="ko",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        compute_input_seq_cache=True,
        save_step=1000,
        save_best_after=1000,
        print_step=25,
        print_eval=True,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
        characters=CharactersConfig(
            characters_class="TTS.tts.models.vits.VitsCharacters",
            pad="<PAD>",
            eos="<EOS>",
            bos="<BOS>",
            blank="<BLNK>",
            characters=" .!?\u1100\u1101\u1102\u1103\u1104\u1105\u1106\u1107\u1108\u1109\u110a\u110b\u110c\u110d\u110e\u110f\u1110\u1111\u1112\u1161\u1162\u1163\u1164\u1165\u1166\u1167\u1168\u1169\u116a\u116b\u116c\u116d\u116e\u116f\u1170\u1171\u1172\u1173\u1174\u1175\u11a8\u11a9\u11aa\u11ab\u11ac\u11ad\u11ae\u11af\u11b0\u11b1\u11b2\u11b3\u11b4\u11b5\u11b6\u11b7\u11b8\u11b9\u11ba\u11bb\u11bc\u11bd\u11be\u11bf\u11c0\u11c1\u11c2",
            punctuations=" .!?",
            phonemes="\u1100\u1101\u1102\u1103\u1104\u1105\u1106\u1107\u1108\u1109\u110a\u110b\u110c\u110d\u110e\u110f\u1110\u1111\u1112\u1161\u1162\u1163\u1164\u1165\u1166\u1167\u1168\u1169\u116a\u116b\u116c\u116d\u116e\u116f\u1170\u1171\u1172\u1173\u1174\u1175\u11a8\u11a9\u11aa\u11ab\u11ac\u11ad\u11ae\u11af\u11b0\u11b1\u11b2\u11b3\u11b4\u11b5\u11b6\u11b7\u11b8\u11b9\u11ba\u11bb\u11bc\u11bd\u11be\u11bf\u11c0\u11c1\u11c2",
        ),
        test_sentences=[
            [
                "아래 문장들은 모델 학습을 위해 사용하지 않은 문장들입니다.",
                "sleeping_ce",
                None,
                "ko",
            ],
            [
                "서울특별시 특허허가과 허가과장 허과장.",
                "sleeping_ce",
                None,
                "ko",
            ],
            [
                "경찰청 철창살은 외철창살이고 검찰청 철창살은 쌍철창살이다.",
                "sleeping_ce",
                None,
                "ko",
            ],
            [
                "지향을 지양으로 오기하는 일을 지양하는 언어 습관을 지향해야 한다.",
                "sleeping_ce",
                None,
                "ko",
            ],
            [
                "그러니까 외계인이 우리 생각을 읽고 우리 생각을 우리가 다시 생각토록 해서 그 생각이 마치 우리가 생각한 것인 것처럼 속였다는 거냐?",
                "sleeping_ce",
                None,
                "ko",
            ],
        ],
    )
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    model = Vits(config, ap, tokenizer, speaker_manager=None)
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    return trainer