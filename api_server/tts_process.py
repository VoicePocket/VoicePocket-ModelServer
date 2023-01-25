import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from module.TTS.TTS.utils.synthesizer import Synthesizer
from text_process import normalize_text

def is_set(email):
    if synth_dict.get(email) == None:
        return False
    return True

def add_synth(email):
    voice_model_path = f"./api_server/voice_model/{email}/"
    # TODO: voice_model_path에 파일이 존재하지 않을 시 버킷에서 다운로드 받는 코드
    
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
    
    # TODO: synthesizer 설정이 끝나면 다운 받은 모델파일을 제거(서버 메모리 문제)

    synth_dict[email] = synthesizer

def make_tts(email, uuid, text):
    wav_path = f'./api_server/wav/{email}_{uuid}.wav'
    syn = synth_dict.get(email)
    text = normalize_text(text)
    wav = syn.tts(text, None, None)
    syn.save_wav(wav, wav_path)
    # TODO: 만든 wav파일을 bucket에 업로드해야 함
    # TODO: 업로드 후 wav파일 제거


synth_dict = {} # syntesizer 객체를 모아두는 딕셔너리