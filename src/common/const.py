# fuck magic number

ROOM_EXPIRE_TIME = 30 * 60  # 30 minutes
ROOM_TOKEN_EXPIRE_TIME = 30 * 60  # 30 minutes
RANDOM_ROOM_EXPIRE_TIME = 5 * 60  # 5 minutes
MAX_RANDOM_ROOM_EXPIRE_TIME = 30 * 60  # 30 minutes
RANDOM_ROOM_TOKEN_EXPIRE_TIME = 5 * 60  # 5 minutes
DAILYLANGCHAINRAGBOT_EXPIRE_TIME = 25 * 60


# VideoFrame type
# https://api-ref.agora.io/en/mediaplayer-kit/windows/1.x/structagora_1_1media_1_1base_1_1_video_frame.html (0-16)
# ExternalVideoFrame format
# https://api-ref.agora.io/en/voice-sdk/unity/4.x/API/enum_videopixelformat.html (0,1,4,16,17)
# video pixel format in Agora SDK
AGORA_VIDEO_PIXEL_UNKNOWN = 0  # The format is known.
AGORA_VIDEO_PIXEL_I420 = 1  # The format is I420. YUV 420
AGORA_VIDEO_PIXEL_BGRA = 2  # The format is BGRA.
AGORA_VIDEO_PIXEL_NV21 = 3  # The format is NV21.
AGORA_VIDEO_PIXEL_RGBA = 4  # The format is RGBA.
AGORA_VIDEO_PIXEL_NV12 = 8  # The format is NV12.
AGORA_VIDEO_TEXTURE_2D = 10  # The format is GL_TEXTURE_2D.
AGORA_VIDEO_TEXTURE_OES = 11  # The format is GL_TEXTURE_OES.
AGORA_VIDEO_PIXEL_I422 = 16  # The format is I422. YUV 422
AGORA_VIDEO_TEXTURE_ID3D11TEXTURE2D = 17

# ExternalVideoFrame type (buffer type)
# https://api-ref.agora.io/en/voice-sdk/unity/4.x/API/enum_videobuffertype.html
# The video buffer in the format
AGORA_VIDEO_BUFFER_RAW_DATA = 1  # raw data.
AGORA_VIDEO_BUFFER_ARRAY = 2  # raw data array.
AGORA_VIDEO_BUFFER_TEXTURE = 3  # Texture.
