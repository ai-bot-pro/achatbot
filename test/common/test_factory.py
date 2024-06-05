
import unittest

from src.common.factory import EngineFactory

r"""
python -m unittest test.common.test_factory
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

    def test_get_engine_by_tag(self):
        from src.modules.speech.asr.whisper_asr import EngineClass
        engine = EngineFactory.get_engine_by_tag(
            EngineClass, "whisper_asr_base")
        self.assertIsInstance(engine, EngineClass)
