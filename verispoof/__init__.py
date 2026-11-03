"""VeriSpoof: one audio language model for speech deepfake detection, source verification and grounded reasoning."""
from verispoof.inference import VeriSpoof, load_audio, parse_reasoning

__all__ = ["VeriSpoof", "load_audio", "parse_reasoning"]
__version__ = "0.1.0"
