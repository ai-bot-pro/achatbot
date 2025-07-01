import logging
import dataclasses

from apipeline.frames.data_frames import Frame
from apipeline.serializers.base_serializer import FrameSerializer

import src.types.frames.protobufs.avatar_data_frames_pb2 as data_frame_protos
from src.types.frames.data_frames import AnimationAudioRawFrame


class AvatarProtobufFrameSerializer(FrameSerializer):
    SERIALIZABLE_TYPES = {
        AnimationAudioRawFrame: "animation_audio",
    }

    SERIALIZABLE_FIELDS = {v: k for k, v in SERIALIZABLE_TYPES.items()}

    def __init__(self):
        pass

    def serialize(self, frame: Frame) -> str | bytes | None:
        proto_frame = data_frame_protos.Frame()
        if type(frame) not in self.SERIALIZABLE_TYPES:
            logging.warning(f"Frame type {type(frame)} is not serializable")
            return None

        # ignoring linter errors; we check that type(frame) is in this dict above
        proto_optional_name = self.SERIALIZABLE_TYPES[type(frame)]
        for field in dataclasses.fields(frame):
            value = getattr(frame, field.name)
            if not value:
                continue

            setattr(getattr(proto_frame, proto_optional_name), field.name, value)

        result = proto_frame.SerializeToString()
        return result

    def deserialize(self, data: str | bytes) -> Frame | None:
        """Returns a Frame object from a Frame protobuf. Used to convert frames
        passed over the wire as protobufs to Frame objects used in pipelines
        and frame processors.

        """

        proto = data_frame_protos.Frame.FromString(data)
        which = proto.WhichOneof("frame")
        if which not in self.SERIALIZABLE_FIELDS:
            logging.error("Unable to deserialize a valid frame")
            return None

        args = getattr(proto, which)
        args_dict = {}
        for field in proto.DESCRIPTOR.fields_by_name[which].message_type.fields:
            args_dict[field.name] = getattr(args, field.name)

        # Remove id name
        if "id" in args_dict:
            del args_dict["id"]
        if "name" in args_dict:
            del args_dict["name"]

        # Create the instance
        class_name = self.SERIALIZABLE_FIELDS[which]
        instance = class_name(**args_dict)

        # Set Frame id name
        if hasattr(args, "id"):
            setattr(instance, "id", getattr(args, "id"))
        if hasattr(args, "name"):
            setattr(instance, "name", getattr(args, "name"))

        return instance


"""
python -m src.serializers.avatar_protobuf
"""
if __name__ == "__main__":
    serializer = AvatarProtobufFrameSerializer()
    src_frame = AnimationAudioRawFrame(audio=b"1234567890", avatar_status="speaking")
    print(src_frame)
    frame = serializer.deserialize(serializer.serialize(src_frame))
    print(frame)
    assert src_frame == frame
