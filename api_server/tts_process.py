import os, sys, tarfile
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from module.TTS.TTS.utils.synthesizer import Synthesizer
from text_process import normalize_text
from bucket_process import down_model_from_bucket, upload_wav_to_bucket

def is_set(email):
    if synth_dict.get(email) == None:
        return False
    return True

def add_synth(email):
    # TODO: best 모델 구성이 결정되면 그에 맞춰 코드를 수정할 것!
    # voice_model_path = f"./api_server/voice_model/{email}_best_model.pth.tar"
    
    # if not os.path.isfile(voice_model_path):
    #     down_model_from_bucket(email)
        
    #     best_model_tar = tarfile.open(voice_model_path)
    #     best_model_tar.extractall(f"./api_server/voice_model")

    synthesizer = Synthesizer(
        "./api_server/voice_model/glowtts/5g_checkpoint.pth.tar",
        "./api_server/voice_model/glowtts/5g_config.json",
        None,
        "./api_server/voice_model/hifigan/5h_checkpoint.pth.tar",
        "./api_server/voice_model/hifigan/5h_config.json",
        None,
        None,
        False,
    )

    synth_dict[email] = synthesizer

    # os.remove(voice_model_path)

def make_tts(email, uuid, text):
    wav_path = f'./api_server/wav/{uuid}.wav'
    syn = synth_dict.get(email)
    symbol = syn.tts_config.characters.characters
    text = normalize_text(text, symbol)
    wav = syn.tts(text, None, None)
    syn.save_wav(wav, wav_path)
    
    upload_wav_to_bucket(wav_path, email, uuid)
    
    os.remove(wav_path)

synth_dict = {} # syntesizer 객체를 모아두는 딕셔너리