import logging
import dataclasses

import apipeline.frames.protobufs.data_frames_pb2 as frame_protos
from apipeline.frames.data_frames import Frame
from apipeline.serializers.protobuf import ProtobufFrameSerializer

import src.types.frames.protobufs.asr_data_frames_pb2 as data_frame_protos
from src.types.frames.data_frames import (
    TextFrame,
    AudioRawFrame,
    ImageRawFrame,
    ASRLiveTranscriptionFrame,
    TranscriptionFrame,
)


class TranscriptionFrameSerializer(ProtobufFrameSerializer):
    SERIALIZABLE_TYPES = {
        ASRLiveTranscriptionFrame: "asr_live_transcription",
        TranscriptionFrame: "transcription",
    }
    SERIALIZABLE_FIELDS = {v: k for k, v in SERIALIZABLE_TYPES.items()}

    def serialize(self, frame: Frame) -> str | bytes | None:
        if type(frame) in ProtobufFrameSerializer.SERIALIZABLE_TYPES:
            return super().serialize(frame)

        if type(frame) not in TranscriptionFrameSerializer.SERIALIZABLE_TYPES:
            logging.warning(f"Frame type {type(frame)} is not serializable")
            return None

        proto_frame = data_frame_protos.Frame()
        # ignoring linter errors; we check that type(frame) is in this dict above
        proto_optional_name = TranscriptionFrameSerializer.SERIALIZABLE_TYPES[type(frame)]
        proto_message = getattr(proto_frame, proto_optional_name)
        for field in dataclasses.fields(frame):
            value = getattr(frame, field.name)
            if not value:
                continue

            if isinstance(value, (list, tuple)):
                getattr(proto_message, field.name).extend(value)
            else:
                setattr(proto_message, field.name, value)

        result = proto_frame.SerializeToString()
        return result

    def deserialize(self, data: str | bytes) -> Frame | None:
        """Returns a Frame object from a Frame protobuf. Used to convert frames
        passed over the wire as protobufs to Frame objects used in pipelines
        and frame processors.

        """
        # text,aduio,image base frame
        proto = frame_protos.Frame.FromString(data)
        which = proto.WhichOneof("frame")
        if which in ProtobufFrameSerializer.SERIALIZABLE_FIELDS:
            return super().deserialize(data)

        # transcription frame
        proto = data_frame_protos.Frame.FromString(data)
        which = proto.WhichOneof("frame")
        if which not in TranscriptionFrameSerializer.SERIALIZABLE_FIELDS:
            logging.error("Unable to deserialize a valid frame")
            return None

        args = getattr(proto, which)
        args_dict = {}
        for field in proto.DESCRIPTOR.fields_by_name[which].message_type.fields:
            if isinstance(getattr(args, field.name), float):
                args_dict[field.name] = round(getattr(args, field.name), 6)
            else:
                args_dict[field.name] = getattr(args, field.name)

        # Remove id name
        if "id" in args_dict:
            del args_dict["id"]
        if "name" in args_dict:
            del args_dict["name"]

        # Create the instance
        class_name = TranscriptionFrameSerializer.SERIALIZABLE_FIELDS[which]
        instance = class_name(**args_dict)

        # Set Frame id name
        if hasattr(args, "id"):
            setattr(instance, "id", getattr(args, "id"))
        if hasattr(args, "name"):
            setattr(instance, "name", getattr(args, "name"))

        return instance


"""
python -m src.serializers.transcription_protobuf
"""
if __name__ == "__main__":
    from src.types.frames import Language

    serializer = TranscriptionFrameSerializer()
    test_cases = [
        TextFrame(text="hello"),
        AudioRawFrame(audio=b"123"),
        ImageRawFrame(
            image=b"321",
            size=(1024, 512),  # width, heigh
            format="JPEG",
            mode="RGB",
        ),
        ASRLiveTranscriptionFrame(
            text="1234567890",
            user_id="uid",
            timestamp="",
            language=Language("zh"),
            timestamps=[123, 321],
            speech_id=100,
            is_final=True,
            start_at_s=round(0.999111, 6),
            cur_at_s=round(2.933, 3),
            end_at_s=round(10.711, 3),
        ),
        TranscriptionFrame(
            text="one two three four five six seven eight nine zero",
            user_id="uid",
            timestamp="2025-08-28T04:48:58.428+00:00",
            language=Language("en"),
            speech_id=110,
            start_at_s=round(0.999111, 6),
            end_at_s=round(10.711, 3),
        ),
    ]
    for src_frame in test_cases:
        print(f"{src_frame=}")
        tgt_frame = serializer.deserialize(serializer.serialize(src_frame))
        print(f"{tgt_frame=}")
        assert src_frame == tgt_frame, f"{src_frame=} != {tgt_frame=}"
        print("PASS!")
