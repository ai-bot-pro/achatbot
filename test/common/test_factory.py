import unittest

from src.common.factory import EngineFactory, EngineClass

r"""
python -m unittest test.common.test_factory.TestEngineFactory.test_get_engines
python -m unittest test.common.test_factory.TestEngineFactory.test_get_engine_by_tag
"""


class TestEngineFactory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_engines(self):
        from src.modules.speech.asr.whisper_asr import IAsr

        engines = EngineFactory.get_engines(IAsr)
        self.assertGreater(len(engines), 0)

        from src.modules.speech.buffering_strategy.none import IBuffering

        engines = EngineFactory.get_engines(IBuffering)
        print(engines)
        self.assertGreater(len(engines), 0)

    def test_get_engine_by_tag(self):
        import src.modules.speech

        stream_engine = EngineFactory.get_engine_by_tag(EngineClass, "pyaudio_stream")
        print(stream_engine)
        self.assertIsInstance(stream_engine, EngineClass)

        stream_engine = EngineFactory.get_engine_by_tag(EngineClass, "daily_room_audio_stream")
        print(stream_engine)
        self.assertIsInstance(stream_engine, EngineClass)

        kwargs = {}
        kwargs["audio_stream"] = stream_engine
        engine = EngineFactory.get_engine_by_tag(EngineClass, "stream_player", **kwargs)
        print(engine)
        self.assertIsInstance(engine, EngineClass)
