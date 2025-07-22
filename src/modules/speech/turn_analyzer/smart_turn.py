from abc import abstractmethod
import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np


try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from transformers import (
        Wav2Vec2Config,
        Wav2Vec2Model,
        Wav2Vec2PreTrainedModel,
        Wav2Vec2Processor,
    )
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    logging.error(
        "In order to use SmartTurnAnalyzerV2, you need to `pip install achatbot[smart_turn]`."
    )
    raise Exception(f"Missing module: {e}")

from src.types.speech.turn_analyzer import EndOfTurnState
from src.types.speech.turn_analyzer.smart_turn import SmartTurnArgs
from src.common.factory import EngineClass
from src.common.interface import ITurnAnalyzer
from src.common.utils.helper import print_model_params


class TurnTimeoutException(Exception):
    """Exception raised when smart turn analysis times out."""

    pass


class BaseSmartTurn(ITurnAnalyzer, EngineClass):
    """Base class for smart turn analyzers using ML models.

    Provides common functionality for smart turn detection including audio
    buffering, speech tracking, and ML model integration. Subclasses must
    implement the specific model prediction logic.
    """

    def __init__(self, **kwargs):
        self.args = SmartTurnArgs(**kwargs)
        # Configuration
        self._stop_ms = self.args.stop_secs * 1000  # silence threshold in ms
        # Inference state
        self._audio_buffer = []
        self._speech_triggered = False
        self._silence_ms = 0
        self._speech_start_time = 0

    @property
    def speech_triggered(self) -> bool:
        """Check if speech has been detected and triggered analysis.

        Returns:
            True if speech has been detected and turn analysis is active.
        """
        return self._speech_triggered

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        """Append audio data for turn analysis.

        Args:
            buffer: Raw audio data bytes to append for analysis.
            is_speech: Whether the audio buffer contains detected speech.

        Returns:
            Current end-of-turn state after processing the audio.
        """
        # Convert raw audio to float32 format and append to the buffer
        audio_int16 = np.frombuffer(buffer, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        self._audio_buffer.append((time.time(), audio_float32))

        state = EndOfTurnState.INCOMPLETE

        if is_speech:
            # Reset silence tracking on speech
            self._silence_ms = 0
            self._speech_triggered = True
            if self._speech_start_time == 0:
                self._speech_start_time = time.time()
        else:
            if self._speech_triggered:
                chunk_duration_ms = len(audio_int16) / (self.args.sample_rate / 1000)
                self._silence_ms += chunk_duration_ms
                # If silence exceeds threshold, mark end of turn
                if self._silence_ms >= self._stop_ms:
                    logging.debug(
                        f"End of Turn complete due to stop_secs. Silence in ms: {self._silence_ms}"
                    )
                    state = EndOfTurnState.COMPLETE
                    self._clear(state)
            else:
                # Trim buffer to prevent unbounded growth before speech
                max_buffer_time = (
                    (self.args.pre_speech_ms / 1000)
                    + self.args.stop_secs
                    + self.args.max_duration_secs
                )
                while (
                    self._audio_buffer and self._audio_buffer[0][0] < time.time() - max_buffer_time
                ):
                    self._audio_buffer.pop(0)

        return state

    async def analyze_end_of_turn(self) -> Tuple[EndOfTurnState, Optional[Dict[str, Any]]]:
        """Analyze the current audio state to determine if turn has ended.

        Returns:
            Tuple containing the end-of-turn state and optional metrics data
            from the ML model analysis.
        """
        state, result = await self._process_speech_segment(self._audio_buffer)
        if state == EndOfTurnState.COMPLETE:
            self._clear(state)
        logging.debug(f"End of Turn result: {state}")
        return state, result

    def clear(self):
        """Reset the turn analyzer to its initial state."""
        self._clear(EndOfTurnState.COMPLETE)

    def _clear(self, turn_state: EndOfTurnState):
        """Clear internal state based on turn completion status."""
        # If the state is still incomplete, keep the _speech_triggered as True
        self._speech_triggered = turn_state == EndOfTurnState.INCOMPLETE
        self._audio_buffer = []
        self._speech_start_time = 0
        self._silence_ms = 0

    async def _process_speech_segment(
        self, audio_buffer
    ) -> Tuple[EndOfTurnState, Optional[Dict[str, Any]]]:
        """Process accumulated audio segment using ML model."""
        state = EndOfTurnState.INCOMPLETE

        if not audio_buffer:
            return state, None

        # Extract recent audio segment for prediction
        start_time = self._speech_start_time - (self.args.pre_speech_ms / 1000)
        start_index = 0
        for i, (t, _) in enumerate(audio_buffer):
            if t >= start_time:
                start_index = i
                break

        end_index = len(audio_buffer) - 1

        # Extract the audio segment
        segment_audio_chunks = [chunk for _, chunk in audio_buffer[start_index : end_index + 1]]
        segment_audio = np.concatenate(segment_audio_chunks)

        # Limit maximum duration
        max_samples = int(self.args.max_duration_secs * self.args.sample_rate)
        if len(segment_audio) > max_samples:
            # slices the array to keep the last max_samples samples, discarding the earlier part.
            segment_audio = segment_audio[-max_samples:]

        result = None

        if len(segment_audio) > 0:
            start_time = time.perf_counter()
            try:
                result = await self._predict_endpoint(segment_audio)
                state = (
                    EndOfTurnState.COMPLETE
                    if result["prediction"] == 1
                    else EndOfTurnState.INCOMPLETE
                )
                end_time = time.perf_counter()

                e2e_processing_time_ms = (end_time - start_time) * 1000

                logging.info(f"{e2e_processing_time_ms=} {state=} {result=}")
            except TurnTimeoutException:
                logging.warning(
                    f"End of Turn complete due to stop_secs. Silence in ms: {self._silence_ms}"
                )
                state = EndOfTurnState.COMPLETE

        else:
            logging.info(f"params: {self.args}, stop_ms: {self._stop_ms}")
            logging.info("Captured empty audio segment, skipping prediction.")

        return state, result

    @abstractmethod
    async def _predict_endpoint(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """Predict end-of-turn using ML model from audio data."""
        pass


class SmartTurnAnalyzerV2(BaseSmartTurn):
    """Local turn analyzer using the smart-turn-v2 PyTorch model.

    Provides end-of-turn detection using locally-stored PyTorch models,
    enabling offline operation without network dependencies. Uses
    Wav2Vec2 architecture for audio sequence classification.
    """

    TAG = "v2_smart_turn_analyzer"

    def __init__(self, **kwargs):
        """Initialize the local PyTorch smart-turn-v2 analyzer.

        Args:
            smart_turn_model_path: Path to directory containing the PyTorch model
                and feature extractor files. If empty, uses default HuggingFace model.
            **kwargs: Additional arguments passed to BaseSmartTurn.
        """
        super().__init__(**kwargs)

        self.torch_dtype = torch.float32
        if self.args.torch_dtype != "auto":
            self.torch_dtype = getattr(torch, self.args.torch_dtype)

        gpu_major = 0
        if torch.cuda.is_available():
            gpu_prop = torch.cuda.get_device_properties("cuda")
            gpu_major = gpu_prop.major

        logging.debug("Loading Local Smart Turn v2 model...")
        # Load the pretrained model for sequence classification
        self._turn_model = Wav2Vec2ForEndpointing.from_pretrained(
            self.args.model_path,
            torch_dtype=self.torch_dtype,
            attn_implementation="flash_attention_2"
            if gpu_major >= 8 and self.torch_dtype == torch.bfloat16
            else None,
        )
        # Load the corresponding feature extractor for preprocessing audio
        self._turn_processor = Wav2Vec2Processor.from_pretrained(self.args.model_path)
        # Set device to GPU if available, else CPU
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Move model to selected device and set it to evaluation mode
        self._turn_model = self._turn_model.to(self._device)
        self._turn_model.eval()
        print_model_params(self._turn_model, extra_info="Smart Turn v2")

        self.warmup()

    def warmup(self):
        if self.args.warmup_steps <= 0:
            return
        # Sample input for tracing (16 seconds of audio at 16kHz)
        audio_random = torch.randn(16000 * 16)  # 16-second dummy audio
        sample_input = self._turn_processor(
            audio_random,
            sampling_rate=16000,
            padding="max_length",
            truncation=True,
            max_length=16000 * 16,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self._device, dtype=self.torch_dtype)

        for i in range(self.args.warmup_steps):
            with torch.no_grad():
                start_time = time.perf_counter()
                _ = self._turn_model(**sample_input)
                end_time = time.perf_counter()
                e2e_processing_time_ms = (end_time - start_time) * 1000
                logging.info(f"warmup step {i} {e2e_processing_time_ms=} ms")

    async def _predict_endpoint(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """Predict end-of-turn using local PyTorch model."""
        inputs = self._turn_processor(
            audio_array,
            sampling_rate=16000,
            padding="max_length",
            truncation=True,
            max_length=16000 * 16,  # 16 seconds at 16kHz
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self._device, dtype=self.torch_dtype)

        # Move inputs to device
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self._turn_model(**inputs)

            # The model returns sigmoid probabilities directly in the logits field
            probability = outputs["logits"][0].item()

            # Make prediction (1 for Complete, 0 for Incomplete)
            prediction = 1 if probability > 0.5 else 0

        return {
            "prediction": prediction,
            "probability": probability,
        }


class Wav2Vec2ForEndpointing(Wav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)

        self.pool_attention = nn.Sequential(
            nn.Linear(config.hidden_size, 256), nn.Tanh(), nn.Linear(256, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        for module in self.classifier:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.1)
                if module.bias is not None:
                    module.bias.data.zero_()

        for module in self.pool_attention:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.1)
                if module.bias is not None:
                    module.bias.data.zero_()

    def attention_pool(self, hidden_states, attention_mask):
        # Calculate attention weights
        attention_weights = self.pool_attention(hidden_states)

        if attention_mask is None:
            raise ValueError("attention_mask must be provided for attention pooling")

        attention_weights = attention_weights + (
            (1.0 - attention_mask.unsqueeze(-1).to(attention_weights.dtype)) * -1e9
        )

        attention_weights = F.softmax(attention_weights, dim=1)

        # Apply attention to hidden states
        weighted_sum = torch.sum(hidden_states * attention_weights, dim=1)

        return weighted_sum

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs[0]

        # Create transformer padding mask
        if attention_mask is not None:
            input_length = attention_mask.size(1)
            hidden_length = hidden_states.size(1)
            ratio = input_length / hidden_length
            indices = (torch.arange(hidden_length, device=attention_mask.device) * ratio).long()
            attention_mask = attention_mask[:, indices]
            attention_mask = attention_mask.bool()
        else:
            attention_mask = None

        pooled = self.attention_pool(hidden_states, attention_mask)

        logits = self.classifier(pooled)

        if torch.isnan(logits).any():
            raise ValueError("NaN values detected in logits")

        probs = torch.sigmoid(logits)
        return {"logits": probs}
