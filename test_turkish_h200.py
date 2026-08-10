import os
import queue
import time
import requests
import sounddevice as sd
import numpy as np

from tts_normalizer import normalize_tts_text


URL = "http://127.0.0.1:8021/tts/stream"


TEXT = (
    "Merhaba Beyzanur, bugün NVIDIA H200 GPU üzerinde çalışan "
    "streaming sesli asistanımızı test ediyoruz. "
    "Sistem 24000 Hz örnekleme hızında çalışıyor ve "
    "ilk cevap yaklaşık 1,5 saniye içinde üretilebiliyorr."
)


TEXT = normalize_tts_text(TEXT)


print(
    "[NORMALIZED TEXT]",
    TEXT
)


SAMPLE_RATE = 24000
CHANNELS = 1


BUFFER_SECONDS = float(
    os.environ.get(
        "TTS_INITIAL_BUFFER_SECONDS",
        "7"
    )
)



class ContinuousAudioPlayer:


    def __init__(self, sample_rate):

        self.sample_rate = sample_rate

        # Sıralı jitter buffer
        self.audio_queue = queue.Queue()

        self.current_chunk = None
        self.current_position = 0

        self.stream = None



    def start(self):


        def callback(
            outdata,
            frames,
            time_info,
            status
        ):


            output = np.zeros(
                frames,
                dtype=np.float32
            )


            filled = 0



            while filled < frames:


                # Yeni chunk gerekiyorsa queue'dan al

                if (
                    self.current_chunk is None
                    or
                    self.current_position >= len(self.current_chunk)
                ):

                    try:

                        self.current_chunk = (
                            self.audio_queue.get(
                                timeout=0.2
                            )
                        )

                        self.current_position = 0


                    except queue.Empty:

                        break



                remaining = (
                    len(self.current_chunk)
                    -
                    self.current_position
                )


                needed = frames - filled


                take = min(
                    remaining,
                    needed
                )



                output[
                    filled:filled + take
                ] = (
                    self.current_chunk[
                        self.current_position:
                        self.current_position + take
                    ]
                )


                self.current_position += take

                filled += take



            outdata[:, 0] = output



            # print(
            #     "QUEUE:",
            #     self.audio_queue.qsize(),
            #     "CURRENT:",
            #     (
            #         len(self.current_chunk)
            #         -
            #         self.current_position
            #         if self.current_chunk is not None
            #         else 0
            #     )
            # )



        self.stream = sd.OutputStream(

            samplerate=self.sample_rate,

            channels=1,

            dtype="float32",

            callback=callback,

            blocksize=2048,

        )


        self.stream.start()




    def add_audio(self, pcm_bytes):


        audio = (
            np.frombuffer(
                pcm_bytes,
                dtype=np.int16
            )
            .astype(np.float32)
            /
            32768.0
        )


        self.audio_queue.put(
            audio
        )




    def stop(self):


        while (

            not self.audio_queue.empty()

            or

            (
                self.current_chunk is not None
                and
                self.current_position < len(self.current_chunk)
            )

        ):

            time.sleep(0.05)



        if self.stream:

            self.stream.stop()

            self.stream.close()




def find_output_device():


    requested = os.environ.get(
        "TTS_OUTPUT_DEVICE"
    )


    if requested:


        for index, device in enumerate(
            sd.query_devices()
        ):


            if (

                device["max_output_channels"] > 0

                and

                requested.lower()
                in device["name"].lower()

            ):

                return index



    return sd.default.device[1]





output_device = find_output_device()



print(
    "Ses çıkışı:",
    sd.query_devices(output_device)["name"]
)



print(
    "Streaming isteği gönderildi..."
)




target_buffer_bytes = int(

    BUFFER_SECONDS
    *
    SAMPLE_RATE
    *
    CHANNELS
    *
    2

)



player = ContinuousAudioPlayer(
    SAMPLE_RATE
)




with requests.post(

    URL,

    json={
        "text": TEXT
    },

    stream=True

) as response:


    response.raise_for_status()



    incoming = response.iter_content(
        chunk_size=16384
    )



    initial_buffer = bytearray()



    # İlk 3 saniyelik buffer

    for audio_bytes in incoming:


        if audio_bytes:

            initial_buffer.extend(
                audio_bytes
            )



        if len(initial_buffer) >= target_buffer_bytes:

            break




    buffered_seconds = (

        len(initial_buffer)

        /

        (SAMPLE_RATE * 2)

    )



    print(
        f"Oynatma başlıyor; "
        f"buffer: {buffered_seconds:.2f}s"
    )




    player.start()



    if initial_buffer:


        player.add_audio(
            bytes(initial_buffer)
        )




    # Stream devam ediyor

    for audio_bytes in incoming:


        if audio_bytes:


            player.add_audio(
                audio_bytes
            )





print(
    "Streaming tamamlandı."
)



player.stop()