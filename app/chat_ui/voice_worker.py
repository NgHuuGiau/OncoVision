from __future__ import annotations

import os
import tempfile
import wave


def build_voice_worker_class(
    *,
    qthread_cls,
    signal_cls,
    np_module,
    pyaudio_module,
    torch_module,
    get_cached_whisper_model,
):
    class VoiceWorker(qthread_cls):
        result_ready = signal_cls(str)
        intensity_changed = signal_cls(int)
        finished = signal_cls()
        error = signal_cls()

        def __init__(self, language: str):
            super().__init__()
            self.lang_code = "vi" if language == "vi" else "en"
            self.is_running = True

        def run(self):
            chunk = 1024
            fs = 16000
            audio = pyaudio_module.PyAudio()
            stream = None
            tmp_path = None
            try:
                stream = audio.open(
                    format=pyaudio_module.paInt16,
                    channels=1,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True,
                )
                frames = []
                silent_chunks = 0
                while self.is_running:
                    data = stream.read(chunk, exception_on_overflow=False)
                    frames.append(data)
                    audio_data = np_module.frombuffer(data, dtype=np_module.int16).astype(np_module.float32)
                    rms = np_module.sqrt(np_module.mean(audio_data**2)) if len(audio_data) > 0 else 0.0
                    intensity = int(min(100, (rms ** 0.65) * 2.2))
                    self.intensity_changed.emit(intensity)
                    if rms < 80:
                        silent_chunks += 1
                    else:
                        silent_chunks = 0
                    if silent_chunks > int(fs / chunk * 1.8):
                        break
                    if len(frames) > int(fs / chunk * 12):
                        break

                if stream and stream.is_active():
                    stream.stop_stream()
                if stream:
                    stream.close()

                if frames:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    tmp_path = tmp.name
                    tmp.close()
                    wav_file = wave.open(tmp_path, "wb")
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(audio.get_sample_size(pyaudio_module.paInt16))
                    wav_file.setframerate(fs)
                    wav_file.writeframes(b"".join(frames))
                    wav_file.close()
                    device = "cuda" if torch_module.cuda.is_available() else "cpu"
                    model = get_cached_whisper_model(language=self.lang_code, device=device)
                    segments, _ = model.transcribe(tmp_path, language=self.lang_code)
                    text = "".join(segment.text for segment in segments)
                    self.result_ready.emit(text.strip())
                else:
                    self.result_ready.emit("")
            except Exception:
                self.error.emit()
            finally:
                try:
                    audio.terminate()
                except Exception:
                    pass
                self.finished.emit()
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

        def stop(self):
            self.is_running = False

    return VoiceWorker
