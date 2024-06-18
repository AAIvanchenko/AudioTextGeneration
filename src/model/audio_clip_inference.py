import sys
import torch

from typing import List, Optional

from pathlib import Path

ROOT = Path(__file__).parents[2]
AUDIO_CLIP_ROOT = Path(ROOT, 'audio_clip')
if str(AUDIO_CLIP_ROOT) not in sys.path:
    sys.path.append(str(AUDIO_CLIP_ROOT))  # add ROOT to PATH

from model.audioclip import AudioCLIP


class AudioCLIPInference(AudioCLIP):
    # Based on https://github.com/AndreyGuzhov/AudioCLIP.git
    def __init__(self,
                 checkpoint: str | Path,
                 return_logit: bool = True,
                 **kwargs
                 ):
        kwargs['pretrained'] = str(checkpoint)
        super().__init__(
            **kwargs
        )
        self.return_logit = return_logit
        self.eval()

    def encode_text(self,
                    text: List[List[str]] | torch.Tensor,
                    base_str: str = '{}',
                    batch_indices: Optional[torch.Tensor] = None) -> torch.Tensor:

        if isinstance(text, list):
            return super(AudioCLIPInference, self).encode_text(text, base_str, batch_indices)

        return super(AudioCLIP, self).encode_text(text)
