import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from .celery_app import wav_worker

@wav_worker.task(name='wav_process')
def wav_process(uuid, email, text):
        from tts_process import add_synth, is_set, make_tts
        if not is_set(email):
            add_synth(email)

        make_tts(email, uuid, text)
    
        return f"{email}/{uuid}.wav"