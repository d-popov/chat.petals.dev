from dataclasses import dataclass
from typing import Optional

import torch

from petals.constants import PUBLIC_INITIAL_PEERS


@dataclass
class ModelInfo:
    repo: str
    adapter: Optional[str] = None
    name: Optional[str] = None


MODELS = [
    # ModelInfo(repo="petals-team/StableBeluga2", name="stabilityai/StableBeluga2"),
    # ModelInfo(repo="tiiuae/falcon-180B-chat"),
    # ModelInfo(repo="WizardLM/WizardCoder-3B-V1.0"),
    ModelInfo(repo="WizardLM/WizardCoder-Python-13B-V1.0"),
    # ModelInfo(repo="WizardLM/WizardCoder-Python-34B-V1.0"),
    ModelInfo(repo="codellama/CodeLlama-34b-Instruct-hf"),
]
DEFAULT_MODEL_NAME = "codellama/CodeLlama-34b-Instruct-hf"

INITIAL_PEERS = PUBLIC_INITIAL_PEERS
# Set this to a list of multiaddrs to connect to a private swarm instead of the public one, for example:
INITIAL_PEERS = ['/ip4/20.101.62.76/tcp/31337/p2p/QmRHP8PQbGCHV7zwBWH2bPipsYfJx44Eo87kdgyteug5Bs','/ip4/92.247.170.10/tcp/31337/p2p/QmSXEQzmS61WcFF6PaFu4NJdwGwtDSPQ67ErGKjTkcTSay']

DEVICE = "cpu"

try:
    from cpufeature import CPUFeature
    has_avx512 = CPUFeature["AVX512f"] and CPUFeature["OS_AVX512"]
except ImportError:
    has_avx512 = False

if DEVICE == "cuda":
    TORCH_DTYPE = "auto"
elif has_avx512:
    TORCH_DTYPE = torch.bfloat16
else:
    TORCH_DTYPE = torch.float32  # You can use bfloat16 in this case too, but it will be slow

STEP_TIMEOUT = 10 * 60
MAX_SESSIONS = 50  # Has effect only for API v1 (HTTP-based)
