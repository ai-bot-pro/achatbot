import wave
from pyogg.opus_encoder import OpusEncoder
from pyogg.opus_decoder import OpusDecoder

if __name__ == "__main__":
    # Setup encoding
    # ==============

    # Read a wav file to obtain PCM data
    filename = "./records/asr_example_zh.wav"
    wave_read = wave.open(filename, "rb")
    print("Reading wav from file '{:s}'".format(filename))

    # Extract the wav's specification
    channels = wave_read.getnchannels()
    print("Number of channels:", channels)
    samples_per_second = wave_read.getframerate()
    print("Sampling frequency:", samples_per_second)
    bytes_per_sample = wave_read.getsampwidth()

    # Create an Opus encoder
    opus_encoder = OpusEncoder()
    opus_encoder.set_application("audio")
    opus_encoder.set_sampling_frequency(samples_per_second)
    opus_encoder.set_channels(channels)

    # Calculate the desired frame size (in samples per channel)
    desired_frame_duration = 20 / 1000  # milliseconds
    desired_frame_size = int(desired_frame_duration * samples_per_second)

    # Setup decoding
    # ==============

    # Create an Opus decoder
    opus_decoder = OpusDecoder()
    opus_decoder.set_channels(channels)
    opus_decoder.set_sampling_frequency(samples_per_second)

    # Open an output wav for the decoded PCM
    output_filename = "./records/out_ogg_asr_example_zh.wav"
    wave_write = wave.open(output_filename, "wb")
    print("Writing wav into file '{:s}'".format(output_filename))

    # Save the wav's specification
    wave_write.setnchannels(channels)
    wave_write.setframerate(samples_per_second)
    wave_write.setsampwidth(bytes_per_sample)

    # Execute encode-decode
    # =====================

    # Loop through the wav file's PCM data and encode it as Opus
    bytes_encoded = 0
    while True:
        # Get data from the wav file
        pcm = wave_read.readframes(desired_frame_size)

        # Check if we've finished reading the wav file
        if len(pcm) == 0:
            break

        # Calculate the effective frame size from the number of bytes
        # read
        effective_frame_size = (
            len(pcm)  # bytes
            // bytes_per_sample
            // channels
        )

        # Check if we've received enough data
        if effective_frame_size < desired_frame_size:
            # We haven't read a full frame from the wav file, so this
            # is most likely a final partial frame before the end of
            # the file.  We'll pad the end of this frame with silence.
            pcm += b"\x00" * (
                (desired_frame_size - effective_frame_size) * bytes_per_sample * channels
            )

        # Encode the PCM data
        encoded_packet = opus_encoder.encode(pcm)
        print(len(encoded_packet.tobytes()), type(encoded_packet))
        bytes_encoded += len(encoded_packet)

        # At this stage we now have a buffer containing an
        # Opus-encoded packet.  This could e sent over UDP, for
        # example, and then decoded with OpusDecoder.  However it
        # cannot really be saved to a file without wrapping it in the
        # likes of an Ogg stream; for this see OggOpusWriter.

        # For this example, we will now immediately decode this
        # encoded packet using OpusDecoder.
        decoded_pcm = opus_decoder.decode(encoded_packet)

        # Save the decoded PCM as a new wav file
        wave_write.writeframes(decoded_pcm)

    wave_read.close()
    wave_write.close()
    print("Total bytes of encoded packets:", bytes_encoded)
    print("Finished.")
