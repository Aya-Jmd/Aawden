"""Tests for security_wsp.prompts: frozen baseline hashes, new wordings, prefill, and regression datasets."""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import re
from pathlib import Path

import pytest

from security_wsp.data import CLEAN, VULNERABLE, load_dataset
from security_wsp.prompts import CONDITIONS, STRONG_VARIANTS, PROMPT_VERSION, prompt_hash, render

ROOT = Path(__file__).resolve().parents[1]

# Hashes recorded from render() BEFORE the wording, code_comment and prefill additions.
# Existing outputs must stay byte-identical: the prompt_hash of every message list is locked here.
BASELINE_CASE_IDS = (
    "c_cab5d4f0e9d4", "c_66cb044697b2", "c_0d4b51222e65", "c_925d109882b9",
)
BASELINE_HASH_VERSION = "aawden-review-v2"
BASELINE_HASHES = {
    ("c_cab5d4f0e9d4", "neutral", False, "reviewer", "trace", False): "2580afe02eeca75b5e176c1508623e0d36ed24872246a64966715406ac6c0791",
    ("c_cab5d4f0e9d4", "neutral", False, "reviewer", "trace", True): "2fc45d6a78ea21bc0658f787d1fc2115697561abef70fac4141f546f1a967e6d",
    ("c_cab5d4f0e9d4", "neutral", False, "reviewer", "verify", False): "2580afe02eeca75b5e176c1508623e0d36ed24872246a64966715406ac6c0791",
    ("c_cab5d4f0e9d4", "neutral", False, "reviewer", "verify", True): "2fc45d6a78ea21bc0658f787d1fc2115697561abef70fac4141f546f1a967e6d",
    ("c_cab5d4f0e9d4", "neutral", False, "authority", "trace", False): "2580afe02eeca75b5e176c1508623e0d36ed24872246a64966715406ac6c0791",
    ("c_cab5d4f0e9d4", "neutral", False, "authority", "trace", True): "2fc45d6a78ea21bc0658f787d1fc2115697561abef70fac4141f546f1a967e6d",
    ("c_cab5d4f0e9d4", "neutral", False, "authority", "verify", False): "2580afe02eeca75b5e176c1508623e0d36ed24872246a64966715406ac6c0791",
    ("c_cab5d4f0e9d4", "neutral", False, "authority", "verify", True): "2fc45d6a78ea21bc0658f787d1fc2115697561abef70fac4141f546f1a967e6d",
    ("c_cab5d4f0e9d4", "neutral", True, "reviewer", "trace", False): "4c6e65377c1ebff54f2d823a88cffc822f7ebc32125649044881e0f3c0bd2777",
    ("c_cab5d4f0e9d4", "neutral", True, "reviewer", "trace", True): "c44221946b0b9cabd19c36ae09d0e0cbed141c5500445a06eef009e063525155",
    ("c_cab5d4f0e9d4", "neutral", True, "reviewer", "verify", False): "ea3c55830fa1d25a2c7ba0106fcd9c02004c8320d3ad9f09bf4d6874f0099610",
    ("c_cab5d4f0e9d4", "neutral", True, "reviewer", "verify", True): "21635b9b1367ee94ceb60e8a04c97415b18895450840b412b590d243ef4eb5f7",
    ("c_cab5d4f0e9d4", "neutral", True, "authority", "trace", False): "4c6e65377c1ebff54f2d823a88cffc822f7ebc32125649044881e0f3c0bd2777",
    ("c_cab5d4f0e9d4", "neutral", True, "authority", "trace", True): "c44221946b0b9cabd19c36ae09d0e0cbed141c5500445a06eef009e063525155",
    ("c_cab5d4f0e9d4", "neutral", True, "authority", "verify", False): "ea3c55830fa1d25a2c7ba0106fcd9c02004c8320d3ad9f09bf4d6874f0099610",
    ("c_cab5d4f0e9d4", "neutral", True, "authority", "verify", True): "21635b9b1367ee94ceb60e8a04c97415b18895450840b412b590d243ef4eb5f7",
    ("c_cab5d4f0e9d4", "correct_suggestion", False, "reviewer", "trace", False): "474f8bb1014a8230671766a344af098bcec41721d70577baba4b0a76cebb4fe6",
    ("c_cab5d4f0e9d4", "correct_suggestion", False, "reviewer", "trace", True): "08fd5654f54d1e8c8a1213543b49abdce3d3afaccbc0b46f1c434b2eecf1d4cf",
    ("c_cab5d4f0e9d4", "correct_suggestion", False, "reviewer", "verify", False): "474f8bb1014a8230671766a344af098bcec41721d70577baba4b0a76cebb4fe6",
    ("c_cab5d4f0e9d4", "correct_suggestion", False, "reviewer", "verify", True): "08fd5654f54d1e8c8a1213543b49abdce3d3afaccbc0b46f1c434b2eecf1d4cf",
    ("c_cab5d4f0e9d4", "correct_suggestion", False, "authority", "trace", False): "536088f117ecf61ee3917fc1ce2a2eee9cac5b889650aa1467779bf53a75eac5",
    ("c_cab5d4f0e9d4", "correct_suggestion", False, "authority", "trace", True): "59982e1ffd02736c1bde9b0af015e1d7e9f361362fdf40681180d3be5ceabdf0",
    ("c_cab5d4f0e9d4", "correct_suggestion", False, "authority", "verify", False): "536088f117ecf61ee3917fc1ce2a2eee9cac5b889650aa1467779bf53a75eac5",
    ("c_cab5d4f0e9d4", "correct_suggestion", False, "authority", "verify", True): "59982e1ffd02736c1bde9b0af015e1d7e9f361362fdf40681180d3be5ceabdf0",
    ("c_cab5d4f0e9d4", "correct_suggestion", True, "reviewer", "trace", False): "3ec4e00b8566eaeaded366ca95bdf608977dd32d8dca510f07f7010f96db4490",
    ("c_cab5d4f0e9d4", "correct_suggestion", True, "reviewer", "trace", True): "f5210f41cba8d1032246bdd39e7671195802c2c35c94595a0099e4773bd8bd38",
    ("c_cab5d4f0e9d4", "correct_suggestion", True, "reviewer", "verify", False): "ce4c9f89946f0e22c657ecc0b216cd144f31ed2a33a15e80842f3eabb9d919e4",
    ("c_cab5d4f0e9d4", "correct_suggestion", True, "reviewer", "verify", True): "bf1cab5dec3d10c1fd488c36b1f2e913762b4b0a8cf4d1d88a70ae2e8db20983",
    ("c_cab5d4f0e9d4", "correct_suggestion", True, "authority", "trace", False): "adc51c33d1899da12ea9b51ed62f5c86522b44c25e469c347317f83173aa07ba",
    ("c_cab5d4f0e9d4", "correct_suggestion", True, "authority", "trace", True): "e8ef6f79db8b9199d97d20d55ea6d1fb7f01c998bab42396bad7c7ccf85fec16",
    ("c_cab5d4f0e9d4", "correct_suggestion", True, "authority", "verify", False): "bec11c05bef74e2bcca037f975207326222dd1b0c893a7d24d2501c0eac70254",
    ("c_cab5d4f0e9d4", "correct_suggestion", True, "authority", "verify", True): "6c1d242f8912abc1d066650f9f794d07bd12034927b06edf90127912464d0a5f",
    ("c_cab5d4f0e9d4", "incorrect_suggestion", False, "reviewer", "trace", False): "ba3e627c0137ac5229ad37a262d3c84dfb6b81d7a44c6803c1a15d1edd7c1db2",
    ("c_cab5d4f0e9d4", "incorrect_suggestion", False, "reviewer", "trace", True): "b6a8139aaf8513d51db8c705efd0857baedd2dbc59b74bbcddfbfe1721541160",
    ("c_cab5d4f0e9d4", "incorrect_suggestion", False, "reviewer", "verify", False): "ba3e627c0137ac5229ad37a262d3c84dfb6b81d7a44c6803c1a15d1edd7c1db2",
    ("c_cab5d4f0e9d4", "incorrect_suggestion", False, "reviewer", "verify", True): "b6a8139aaf8513d51db8c705efd0857baedd2dbc59b74bbcddfbfe1721541160",
    ("c_cab5d4f0e9d4", "incorrect_suggestion", False, "authority", "trace", False): "19062cfe79c388ae74f28ee818ba9c2552677d6137f5b8b207c873461c119fe7",
    ("c_cab5d4f0e9d4", "incorrect_suggestion", False, "authority", "trace", True): "d86dc42ade9801e1360b47fef5efeee54b1362c33337fa95f7a1118517096432",
    ("c_cab5d4f0e9d4", "incorrect_suggestion", False, "authority", "verify", False): "19062cfe79c388ae74f28ee818ba9c2552677d6137f5b8b207c873461c119fe7",
    ("c_cab5d4f0e9d4", "incorrect_suggestion", False, "authority", "verify", True): "d86dc42ade9801e1360b47fef5efeee54b1362c33337fa95f7a1118517096432",
    ("c_cab5d4f0e9d4", "incorrect_suggestion", True, "reviewer", "trace", False): "548c33b39e6f65fd615601764ab62d6e4e1f2a09b42205da894c1b0ec6b47a39",
    ("c_cab5d4f0e9d4", "incorrect_suggestion", True, "reviewer", "trace", True): "16198760508846b6633b00d6d7b515ce01705948d00c3bcab1ff4ab653e5d1d4",
    ("c_cab5d4f0e9d4", "incorrect_suggestion", True, "reviewer", "verify", False): "b29e0dbc8966d04d938176ac7f49d69bf8fc1ed76e5c150c04881ea57b3dbb77",
    ("c_cab5d4f0e9d4", "incorrect_suggestion", True, "reviewer", "verify", True): "a3b89586f8076afa56814802249b6365310e307ea0aa6d54bb88474a2f61c9fe",
    ("c_cab5d4f0e9d4", "incorrect_suggestion", True, "authority", "trace", False): "9dbfd069ec408abf8f8cca756d92b56da60037e4f1f66c86e1aad70943a36dce",
    ("c_cab5d4f0e9d4", "incorrect_suggestion", True, "authority", "trace", True): "68a60a4de256d1f254294bb84a1b19f044cb434c1c322cada46f92dce3f6bd06",
    ("c_cab5d4f0e9d4", "incorrect_suggestion", True, "authority", "verify", False): "ef4f4e93d6cecee3ed8e8d742f43a25aae2ad08eef2101be1611c97d410f9bcf",
    ("c_cab5d4f0e9d4", "incorrect_suggestion", True, "authority", "verify", True): "cfabe2f98ea264d6cbfd0ba1b503a9af999cec43b044df94fdfde5af8a2ac28f",
    ("c_66cb044697b2", "neutral", False, "reviewer", "trace", False): "16628405c3def88e851bb28940b6fc856087cbb133d7b1057c0128ece2a6d2fb",
    ("c_66cb044697b2", "neutral", False, "reviewer", "trace", True): "e17d1d98c4bb68139719b057097a46e0f71c1d44030d6f3d40f0b27b3e8906c5",
    ("c_66cb044697b2", "neutral", False, "reviewer", "verify", False): "16628405c3def88e851bb28940b6fc856087cbb133d7b1057c0128ece2a6d2fb",
    ("c_66cb044697b2", "neutral", False, "reviewer", "verify", True): "e17d1d98c4bb68139719b057097a46e0f71c1d44030d6f3d40f0b27b3e8906c5",
    ("c_66cb044697b2", "neutral", False, "authority", "trace", False): "16628405c3def88e851bb28940b6fc856087cbb133d7b1057c0128ece2a6d2fb",
    ("c_66cb044697b2", "neutral", False, "authority", "trace", True): "e17d1d98c4bb68139719b057097a46e0f71c1d44030d6f3d40f0b27b3e8906c5",
    ("c_66cb044697b2", "neutral", False, "authority", "verify", False): "16628405c3def88e851bb28940b6fc856087cbb133d7b1057c0128ece2a6d2fb",
    ("c_66cb044697b2", "neutral", False, "authority", "verify", True): "e17d1d98c4bb68139719b057097a46e0f71c1d44030d6f3d40f0b27b3e8906c5",
    ("c_66cb044697b2", "neutral", True, "reviewer", "trace", False): "ad4f896d19dd486df6d5a705607d7101eb5ad6a6506632b811fb475696d3eae4",
    ("c_66cb044697b2", "neutral", True, "reviewer", "trace", True): "eef1d37a0c32d356d56f55c50e86df35a1c58100ce9dd130724b49f88e9f272c",
    ("c_66cb044697b2", "neutral", True, "reviewer", "verify", False): "a70046083d0646864bab4b5cf8a19919af062d5e0ad4b70d571a63914a51591c",
    ("c_66cb044697b2", "neutral", True, "reviewer", "verify", True): "adfc1d80bb5cbdd6e7157e54764d3dde9cc376e915e346f9b4e3441b4e3fc6b3",
    ("c_66cb044697b2", "neutral", True, "authority", "trace", False): "ad4f896d19dd486df6d5a705607d7101eb5ad6a6506632b811fb475696d3eae4",
    ("c_66cb044697b2", "neutral", True, "authority", "trace", True): "eef1d37a0c32d356d56f55c50e86df35a1c58100ce9dd130724b49f88e9f272c",
    ("c_66cb044697b2", "neutral", True, "authority", "verify", False): "a70046083d0646864bab4b5cf8a19919af062d5e0ad4b70d571a63914a51591c",
    ("c_66cb044697b2", "neutral", True, "authority", "verify", True): "adfc1d80bb5cbdd6e7157e54764d3dde9cc376e915e346f9b4e3441b4e3fc6b3",
    ("c_66cb044697b2", "correct_suggestion", False, "reviewer", "trace", False): "4dcba087bade7242d8ac5b3d48ae44c35bf872ed47cba739c770b3537f2ff55c",
    ("c_66cb044697b2", "correct_suggestion", False, "reviewer", "trace", True): "646e133a7dc5b79c55a89b59b0b0a3894a5128b9f36184f140e1a80c904b602a",
    ("c_66cb044697b2", "correct_suggestion", False, "reviewer", "verify", False): "4dcba087bade7242d8ac5b3d48ae44c35bf872ed47cba739c770b3537f2ff55c",
    ("c_66cb044697b2", "correct_suggestion", False, "reviewer", "verify", True): "646e133a7dc5b79c55a89b59b0b0a3894a5128b9f36184f140e1a80c904b602a",
    ("c_66cb044697b2", "correct_suggestion", False, "authority", "trace", False): "58643ff5331ca147e6dbf3871c6c83d85a38d9408df300bf5b9ea7403c8b90ce",
    ("c_66cb044697b2", "correct_suggestion", False, "authority", "trace", True): "548024e011d22c73970e6b7d401e0fa2950cac74f2e1e683cb22fd397d7dbcab",
    ("c_66cb044697b2", "correct_suggestion", False, "authority", "verify", False): "58643ff5331ca147e6dbf3871c6c83d85a38d9408df300bf5b9ea7403c8b90ce",
    ("c_66cb044697b2", "correct_suggestion", False, "authority", "verify", True): "548024e011d22c73970e6b7d401e0fa2950cac74f2e1e683cb22fd397d7dbcab",
    ("c_66cb044697b2", "correct_suggestion", True, "reviewer", "trace", False): "b989cc60cd49b263ce63613966fbbb7d98e7585868923cd1a394db5fe4de282a",
    ("c_66cb044697b2", "correct_suggestion", True, "reviewer", "trace", True): "3423d5971cfd234c476cc840bf47695781b87770d31b48bf2bd447e7240d966c",
    ("c_66cb044697b2", "correct_suggestion", True, "reviewer", "verify", False): "a5478a2023d26bcdced78c0492c0267d52364ecf5a19a8b1e225db58a410f19a",
    ("c_66cb044697b2", "correct_suggestion", True, "reviewer", "verify", True): "60f6b1e99f5855d2b5949a797686f90c21fd449ea888d5d9a00af3277ca8d77c",
    ("c_66cb044697b2", "correct_suggestion", True, "authority", "trace", False): "5f527dd2641d6d3822316afefd21e4e2b18fb8f2eadda9c5841111a98f145a7f",
    ("c_66cb044697b2", "correct_suggestion", True, "authority", "trace", True): "7d87fa12a6c559885b41d20b60860641fec5f78aacbe955907ec2bdf30e693e0",
    ("c_66cb044697b2", "correct_suggestion", True, "authority", "verify", False): "f570144b1542ed3de11ab04b1ab069d6e120f0d37a734b26116917aa045ab447",
    ("c_66cb044697b2", "correct_suggestion", True, "authority", "verify", True): "17993508ab00f63a23b1f65a1a8d87b164cb74ac2cbd1a2fa866d726f493a5f3",
    ("c_66cb044697b2", "incorrect_suggestion", False, "reviewer", "trace", False): "432933ce7e0a0290803b5cd96153ce85b22696b46ebf0a64ac96d235ad1eff71",
    ("c_66cb044697b2", "incorrect_suggestion", False, "reviewer", "trace", True): "4bf2fe890975f7a7cfd580cb38360c256ec62cf797ee1f01de5c3277e93276d6",
    ("c_66cb044697b2", "incorrect_suggestion", False, "reviewer", "verify", False): "432933ce7e0a0290803b5cd96153ce85b22696b46ebf0a64ac96d235ad1eff71",
    ("c_66cb044697b2", "incorrect_suggestion", False, "reviewer", "verify", True): "4bf2fe890975f7a7cfd580cb38360c256ec62cf797ee1f01de5c3277e93276d6",
    ("c_66cb044697b2", "incorrect_suggestion", False, "authority", "trace", False): "6237334268def717f644edeca3d5122b4fc5e6a582a32488f30c9195da029b8f",
    ("c_66cb044697b2", "incorrect_suggestion", False, "authority", "trace", True): "276cb17321f95fde8b513ce2c4aa6538dd5d9b942d84ee595fc3feacf78f9d36",
    ("c_66cb044697b2", "incorrect_suggestion", False, "authority", "verify", False): "6237334268def717f644edeca3d5122b4fc5e6a582a32488f30c9195da029b8f",
    ("c_66cb044697b2", "incorrect_suggestion", False, "authority", "verify", True): "276cb17321f95fde8b513ce2c4aa6538dd5d9b942d84ee595fc3feacf78f9d36",
    ("c_66cb044697b2", "incorrect_suggestion", True, "reviewer", "trace", False): "50f890b7d307dd7e623906b094de83b444db99935efaa4771fd6ced16e8ccac0",
    ("c_66cb044697b2", "incorrect_suggestion", True, "reviewer", "trace", True): "73ab7558aff3de9a0ec714b3b8ef14f4dcb4be782458d951bbc930e90f89ba75",
    ("c_66cb044697b2", "incorrect_suggestion", True, "reviewer", "verify", False): "e9815ea98c784319d71e6acd84cc039ca99bbda118eea7910fbb5974661c3b71",
    ("c_66cb044697b2", "incorrect_suggestion", True, "reviewer", "verify", True): "dd8d2fc19cc1eade1acc3c42ca61e16e95db4ab644cff01f6be9e07b2929f6d7",
    ("c_66cb044697b2", "incorrect_suggestion", True, "authority", "trace", False): "cce80175b5f4ec41d599307973821474de87ae7de12ba19640bfdcf9bd1da3cf",
    ("c_66cb044697b2", "incorrect_suggestion", True, "authority", "trace", True): "f0cb001669600ffd47d0db6c833d3cce9733c95f4d9ae96f6a5114257b8ea643",
    ("c_66cb044697b2", "incorrect_suggestion", True, "authority", "verify", False): "fc53df5e8fb65d83e95bb5e05b247df55ccb6d846a872a1832d4f421c9dd9c3d",
    ("c_66cb044697b2", "incorrect_suggestion", True, "authority", "verify", True): "ec674a84c949f0b64bb1a283ccd0663bdacbe9dac944d8abeceb6a09f6ec62fe",
    ("c_0d4b51222e65", "neutral", False, "reviewer", "trace", False): "84e457e66e5b21c3d61a9709366e8a0a1344197bcd21cc42c5cf04c8f58c148c",
    ("c_0d4b51222e65", "neutral", False, "reviewer", "trace", True): "6a5080f7f080fe1ed2d0b35a9b2111f0b57af98359f9ddee01c2f7f30abf7a30",
    ("c_0d4b51222e65", "neutral", False, "reviewer", "verify", False): "84e457e66e5b21c3d61a9709366e8a0a1344197bcd21cc42c5cf04c8f58c148c",
    ("c_0d4b51222e65", "neutral", False, "reviewer", "verify", True): "6a5080f7f080fe1ed2d0b35a9b2111f0b57af98359f9ddee01c2f7f30abf7a30",
    ("c_0d4b51222e65", "neutral", False, "authority", "trace", False): "84e457e66e5b21c3d61a9709366e8a0a1344197bcd21cc42c5cf04c8f58c148c",
    ("c_0d4b51222e65", "neutral", False, "authority", "trace", True): "6a5080f7f080fe1ed2d0b35a9b2111f0b57af98359f9ddee01c2f7f30abf7a30",
    ("c_0d4b51222e65", "neutral", False, "authority", "verify", False): "84e457e66e5b21c3d61a9709366e8a0a1344197bcd21cc42c5cf04c8f58c148c",
    ("c_0d4b51222e65", "neutral", False, "authority", "verify", True): "6a5080f7f080fe1ed2d0b35a9b2111f0b57af98359f9ddee01c2f7f30abf7a30",
    ("c_0d4b51222e65", "neutral", True, "reviewer", "trace", False): "a1f36f022f1009b92dbb84783b4f97ed68a63919f6166806b27403e2afced928",
    ("c_0d4b51222e65", "neutral", True, "reviewer", "trace", True): "92cd466875dc0210755147054f0d096adb6f4ec52788a23f28929493a4fd3b1e",
    ("c_0d4b51222e65", "neutral", True, "reviewer", "verify", False): "dd2c313763080c4e0245c56a691bb071ab995910320460508d9b1f1f39644d62",
    ("c_0d4b51222e65", "neutral", True, "reviewer", "verify", True): "46acdbd84048371ba628186f4ba5d89399bc51ca216fdcd95dda86be96e2164a",
    ("c_0d4b51222e65", "neutral", True, "authority", "trace", False): "a1f36f022f1009b92dbb84783b4f97ed68a63919f6166806b27403e2afced928",
    ("c_0d4b51222e65", "neutral", True, "authority", "trace", True): "92cd466875dc0210755147054f0d096adb6f4ec52788a23f28929493a4fd3b1e",
    ("c_0d4b51222e65", "neutral", True, "authority", "verify", False): "dd2c313763080c4e0245c56a691bb071ab995910320460508d9b1f1f39644d62",
    ("c_0d4b51222e65", "neutral", True, "authority", "verify", True): "46acdbd84048371ba628186f4ba5d89399bc51ca216fdcd95dda86be96e2164a",
    ("c_0d4b51222e65", "correct_suggestion", False, "reviewer", "trace", False): "25fb67e8b7116c06b5b5ff60fa6cf9e96f90c93fc2231c163f221d8d5eb8d279",
    ("c_0d4b51222e65", "correct_suggestion", False, "reviewer", "trace", True): "457ee5e4e3f02802d837f4e01ae964cc27f76843e068785ac5cfc735bcc23e3e",
    ("c_0d4b51222e65", "correct_suggestion", False, "reviewer", "verify", False): "25fb67e8b7116c06b5b5ff60fa6cf9e96f90c93fc2231c163f221d8d5eb8d279",
    ("c_0d4b51222e65", "correct_suggestion", False, "reviewer", "verify", True): "457ee5e4e3f02802d837f4e01ae964cc27f76843e068785ac5cfc735bcc23e3e",
    ("c_0d4b51222e65", "correct_suggestion", False, "authority", "trace", False): "c4e913fe72c90413b8f3d5937178d9dab364f5695ee756e14de10ea8331ec991",
    ("c_0d4b51222e65", "correct_suggestion", False, "authority", "trace", True): "d46d9963837fc17f9c6eabf664ef8a8c762f94f1fe84f1f37467371ee80c24c3",
    ("c_0d4b51222e65", "correct_suggestion", False, "authority", "verify", False): "c4e913fe72c90413b8f3d5937178d9dab364f5695ee756e14de10ea8331ec991",
    ("c_0d4b51222e65", "correct_suggestion", False, "authority", "verify", True): "d46d9963837fc17f9c6eabf664ef8a8c762f94f1fe84f1f37467371ee80c24c3",
    ("c_0d4b51222e65", "correct_suggestion", True, "reviewer", "trace", False): "dd78f3739ccac60639128fb4a347971c52ca894d62e08086753d60c8a28dba50",
    ("c_0d4b51222e65", "correct_suggestion", True, "reviewer", "trace", True): "06793de15f9bf83f596094202b45c4af3557b381f808bc5bc48f9a897f531951",
    ("c_0d4b51222e65", "correct_suggestion", True, "reviewer", "verify", False): "96f004a72dd5aa3ece478519daac34acd5dd7be05f76ef218414fd3501534a6a",
    ("c_0d4b51222e65", "correct_suggestion", True, "reviewer", "verify", True): "ec923b9e743f5820f7617bbaa33675fce3cea7b5cf65872b9026012175489346",
    ("c_0d4b51222e65", "correct_suggestion", True, "authority", "trace", False): "efdb50db892835a4af818610058a3e2356f68b58a41ea3ba692129629fc046fc",
    ("c_0d4b51222e65", "correct_suggestion", True, "authority", "trace", True): "00a37b4d97aa9b5eafbbf837de735683009b3be586bb5b54f3d2afcdd63dc0b1",
    ("c_0d4b51222e65", "correct_suggestion", True, "authority", "verify", False): "a63d9fbc00c87dea221b0923613e339b5e644fb5f356539521ddbf3c103d4761",
    ("c_0d4b51222e65", "correct_suggestion", True, "authority", "verify", True): "25d93e5c0de5f657958e2b21f2bb4b299f707246b6d8e8756b9a7df6d8fff34f",
    ("c_0d4b51222e65", "incorrect_suggestion", False, "reviewer", "trace", False): "0fe5c050681c4bdbe97dbbc69ec64c926b78cb00c6fd9a02e238900cd1168073",
    ("c_0d4b51222e65", "incorrect_suggestion", False, "reviewer", "trace", True): "c92de02e97f5c4d2ea9c9adacf1c6fe4c591515c3f9f037d1ea8c4dde3a8f019",
    ("c_0d4b51222e65", "incorrect_suggestion", False, "reviewer", "verify", False): "0fe5c050681c4bdbe97dbbc69ec64c926b78cb00c6fd9a02e238900cd1168073",
    ("c_0d4b51222e65", "incorrect_suggestion", False, "reviewer", "verify", True): "c92de02e97f5c4d2ea9c9adacf1c6fe4c591515c3f9f037d1ea8c4dde3a8f019",
    ("c_0d4b51222e65", "incorrect_suggestion", False, "authority", "trace", False): "f1896211b87b48bf141ba1ee3f865a2f870f15d5e1d5b2caae827d4d1d12c84c",
    ("c_0d4b51222e65", "incorrect_suggestion", False, "authority", "trace", True): "0497828d23bed270309c4c24667028530f1af5dd1c3b5a6b02808bfb1b1185fb",
    ("c_0d4b51222e65", "incorrect_suggestion", False, "authority", "verify", False): "f1896211b87b48bf141ba1ee3f865a2f870f15d5e1d5b2caae827d4d1d12c84c",
    ("c_0d4b51222e65", "incorrect_suggestion", False, "authority", "verify", True): "0497828d23bed270309c4c24667028530f1af5dd1c3b5a6b02808bfb1b1185fb",
    ("c_0d4b51222e65", "incorrect_suggestion", True, "reviewer", "trace", False): "acce515b2e0d6f37d723532a7a1301ce0a54470068df54842acd7946ea94275a",
    ("c_0d4b51222e65", "incorrect_suggestion", True, "reviewer", "trace", True): "347c7146705dc5f374e7126a45548193541ad9d8aa32fba48690323820034e50",
    ("c_0d4b51222e65", "incorrect_suggestion", True, "reviewer", "verify", False): "b3a634e46fabe177e91a228ca0962097b627573872c4b5457aab5fa70310b9bb",
    ("c_0d4b51222e65", "incorrect_suggestion", True, "reviewer", "verify", True): "3e5424a62803df45c3b4acc39b94187c254e4de42c1e1984fff7991fe5e815f1",
    ("c_0d4b51222e65", "incorrect_suggestion", True, "authority", "trace", False): "fc041eca25f602374160806ed8e3b2837869560f511beb1e293bd1b0ed095ec8",
    ("c_0d4b51222e65", "incorrect_suggestion", True, "authority", "trace", True): "be975aabba7ae5e22a7eca8388d2d895e1bb4cf26fecf9a6d7f4e53ceb950c7f",
    ("c_0d4b51222e65", "incorrect_suggestion", True, "authority", "verify", False): "b142f18e7489d3075578526dfe88c0aa10ebdc0809b426e2f809f7775087a5ff",
    ("c_0d4b51222e65", "incorrect_suggestion", True, "authority", "verify", True): "1b7ab305885fb534c71003e1b0032688bb7823748b4c17b56751be2c38e4f511",
    ("c_925d109882b9", "neutral", False, "reviewer", "trace", False): "6c2b891529c5b5f3da6d6a4e014a09d3dffc8dd7cc91ce868048e7d0843a7545",
    ("c_925d109882b9", "neutral", False, "reviewer", "trace", True): "9d51a6a5b296bdf10cd129c734329ba8009150f5d0b9f9a4cfad5b92fcb27436",
    ("c_925d109882b9", "neutral", False, "reviewer", "verify", False): "6c2b891529c5b5f3da6d6a4e014a09d3dffc8dd7cc91ce868048e7d0843a7545",
    ("c_925d109882b9", "neutral", False, "reviewer", "verify", True): "9d51a6a5b296bdf10cd129c734329ba8009150f5d0b9f9a4cfad5b92fcb27436",
    ("c_925d109882b9", "neutral", False, "authority", "trace", False): "6c2b891529c5b5f3da6d6a4e014a09d3dffc8dd7cc91ce868048e7d0843a7545",
    ("c_925d109882b9", "neutral", False, "authority", "trace", True): "9d51a6a5b296bdf10cd129c734329ba8009150f5d0b9f9a4cfad5b92fcb27436",
    ("c_925d109882b9", "neutral", False, "authority", "verify", False): "6c2b891529c5b5f3da6d6a4e014a09d3dffc8dd7cc91ce868048e7d0843a7545",
    ("c_925d109882b9", "neutral", False, "authority", "verify", True): "9d51a6a5b296bdf10cd129c734329ba8009150f5d0b9f9a4cfad5b92fcb27436",
    ("c_925d109882b9", "neutral", True, "reviewer", "trace", False): "7c7ae8d17325fc23c5e29db97c6a94c0b9b2bc793512654c15df5a6847cebdf3",
    ("c_925d109882b9", "neutral", True, "reviewer", "trace", True): "d77b8352cd2ea8d1ea5a7e8c3d44921ee8136d0c11348d7e893635f1251c9311",
    ("c_925d109882b9", "neutral", True, "reviewer", "verify", False): "b59b3d75599bd5008f8a88e44374b05abbdfbc793ab49cfc14c720ffb674cf86",
    ("c_925d109882b9", "neutral", True, "reviewer", "verify", True): "3a369ab66e73411b8c355d60be8f16a821eb326b818987e84b1c354a7d4a98c5",
    ("c_925d109882b9", "neutral", True, "authority", "trace", False): "7c7ae8d17325fc23c5e29db97c6a94c0b9b2bc793512654c15df5a6847cebdf3",
    ("c_925d109882b9", "neutral", True, "authority", "trace", True): "d77b8352cd2ea8d1ea5a7e8c3d44921ee8136d0c11348d7e893635f1251c9311",
    ("c_925d109882b9", "neutral", True, "authority", "verify", False): "b59b3d75599bd5008f8a88e44374b05abbdfbc793ab49cfc14c720ffb674cf86",
    ("c_925d109882b9", "neutral", True, "authority", "verify", True): "3a369ab66e73411b8c355d60be8f16a821eb326b818987e84b1c354a7d4a98c5",
    ("c_925d109882b9", "correct_suggestion", False, "reviewer", "trace", False): "931fca85e60064e1ec97e0cc3dcc5e96b5f30161e632dafde24af2bc8d25048a",
    ("c_925d109882b9", "correct_suggestion", False, "reviewer", "trace", True): "a26f41636871f6117ca460f717992bb03035021ec8ec09f22543344251996293",
    ("c_925d109882b9", "correct_suggestion", False, "reviewer", "verify", False): "931fca85e60064e1ec97e0cc3dcc5e96b5f30161e632dafde24af2bc8d25048a",
    ("c_925d109882b9", "correct_suggestion", False, "reviewer", "verify", True): "a26f41636871f6117ca460f717992bb03035021ec8ec09f22543344251996293",
    ("c_925d109882b9", "correct_suggestion", False, "authority", "trace", False): "581902dce400d71e2fc500c85e5f82b19e130b6fb67958268e2eb3f586ec9c55",
    ("c_925d109882b9", "correct_suggestion", False, "authority", "trace", True): "f2bda11aa091cfb1ba91bf63aaabed42307d5a252b0e6b3067d78f248b582140",
    ("c_925d109882b9", "correct_suggestion", False, "authority", "verify", False): "581902dce400d71e2fc500c85e5f82b19e130b6fb67958268e2eb3f586ec9c55",
    ("c_925d109882b9", "correct_suggestion", False, "authority", "verify", True): "f2bda11aa091cfb1ba91bf63aaabed42307d5a252b0e6b3067d78f248b582140",
    ("c_925d109882b9", "correct_suggestion", True, "reviewer", "trace", False): "af01aca3f9bd15992c99593e4d2d5663a6d68459223b972d04107102c29332ec",
    ("c_925d109882b9", "correct_suggestion", True, "reviewer", "trace", True): "f38d2455f72f06462e79fc08bdb3fd910f5e264ca7cfae6c2e94948369ad8906",
    ("c_925d109882b9", "correct_suggestion", True, "reviewer", "verify", False): "89a64fb84df4b963d28cdababdc4d465e4927e5159126e771177ca8da3472266",
    ("c_925d109882b9", "correct_suggestion", True, "reviewer", "verify", True): "458a7b746dd4380f98444e0c2dfab2f11aff4e19129717e29c9cf614e27047d7",
    ("c_925d109882b9", "correct_suggestion", True, "authority", "trace", False): "3fcb39e766e457297e9342fba3ca1fb95c70cf2066ce1698fa0aaebdc0834417",
    ("c_925d109882b9", "correct_suggestion", True, "authority", "trace", True): "f6897a1f3cdf0fb6cdf43d2f1adc79bff5325da95eed78582d46dde2a3e3d007",
    ("c_925d109882b9", "correct_suggestion", True, "authority", "verify", False): "670f2ff8e97f379e83361ace6ce8d4a83acefa6db6b137a78950fa79c5d6a3c7",
    ("c_925d109882b9", "correct_suggestion", True, "authority", "verify", True): "3ac64666aa1072ef369c9affad26f7f447a392b2193097be556221390f9ac650",
    ("c_925d109882b9", "incorrect_suggestion", False, "reviewer", "trace", False): "97c450d5ea308ec7ace3f781b5eb0549da0de2472e297cc5b36a875585a9dbc0",
    ("c_925d109882b9", "incorrect_suggestion", False, "reviewer", "trace", True): "192ec35b888851b5ead7a7c007b2485491e6f89325c2870c8b2b39f49b8f8322",
    ("c_925d109882b9", "incorrect_suggestion", False, "reviewer", "verify", False): "97c450d5ea308ec7ace3f781b5eb0549da0de2472e297cc5b36a875585a9dbc0",
    ("c_925d109882b9", "incorrect_suggestion", False, "reviewer", "verify", True): "192ec35b888851b5ead7a7c007b2485491e6f89325c2870c8b2b39f49b8f8322",
    ("c_925d109882b9", "incorrect_suggestion", False, "authority", "trace", False): "1ac7ff8444741d342fedcf574e6786f8bf0dd864f285467f12142bb3477e1c28",
    ("c_925d109882b9", "incorrect_suggestion", False, "authority", "trace", True): "2994381da71106d861c08dbb14c65a12018707316182b94004f2e0432c706a8d",
    ("c_925d109882b9", "incorrect_suggestion", False, "authority", "verify", False): "1ac7ff8444741d342fedcf574e6786f8bf0dd864f285467f12142bb3477e1c28",
    ("c_925d109882b9", "incorrect_suggestion", False, "authority", "verify", True): "2994381da71106d861c08dbb14c65a12018707316182b94004f2e0432c706a8d",
    ("c_925d109882b9", "incorrect_suggestion", True, "reviewer", "trace", False): "f70bf8a080425668748227773a6ff95e942e15a2996c808fbd3331d3f73b004b",
    ("c_925d109882b9", "incorrect_suggestion", True, "reviewer", "trace", True): "f0491cf55de98bb7238c40620c34092eb86434c7d90714a8b87ce18e404ca715",
    ("c_925d109882b9", "incorrect_suggestion", True, "reviewer", "verify", False): "7a631601bc6014cfa4a93d51198c1195b320eed31d10d43e7777ecb786b87a30",
    ("c_925d109882b9", "incorrect_suggestion", True, "reviewer", "verify", True): "59b8fc1eaa88b548fceb04fbe48cf09aaccf20738a59679738782c4dc047e8f8",
    ("c_925d109882b9", "incorrect_suggestion", True, "authority", "trace", False): "0849dd6c980e2174e398f4f23be87b40684a5b4023161e425095dd81e7168256",
    ("c_925d109882b9", "incorrect_suggestion", True, "authority", "trace", True): "c23bcc7a0a168d02a3d0e056794ce8c48fb071df3afb276e498b64d0a33a7d4a",
    ("c_925d109882b9", "incorrect_suggestion", True, "authority", "verify", False): "1f47e0477f7237ba28fd8694e643ef9f9e341667bcdafab65a826a1ceadc7562",
    ("c_925d109882b9", "incorrect_suggestion", True, "authority", "verify", True): "4610a9d5fb748b90ab125e31c9a9b79b80a6b911060e561e6e7d99b0d0f7c210",
}


def screen_cases():
    return {case.case_id: case for case in load_dataset(ROOT / "datasets" / "screen.json")}


def test_prompt_version_unchanged():
    assert PROMPT_VERSION == BASELINE_HASH_VERSION


def test_existing_prompt_hashes_are_byte_identical():
    cases = screen_cases()
    assert set(BASELINE_CASE_IDS) <= set(cases)
    mismatches = {}
    for key, expected in BASELINE_HASHES.items():
        cid, condition, strong, wording, variant, numbered = key
        actual = prompt_hash(render(cases[cid], condition, strong, wording=wording,
                                    strong_variant=variant, numbered_source=numbered))
        if actual != expected:
            mismatches[key] = actual
    assert mismatches == {}


def test_baseline_covers_every_existing_combination():
    combos = {(c, cond, s, w, v, n) for c in BASELINE_CASE_IDS for cond in CONDITIONS
              for s in (False, True) for w in ("reviewer", "authority") for v in STRONG_VARIANTS
              for n in (False, True)}
    assert set(BASELINE_HASHES) == combos


NEW_WORDINGS = {
    "ciso": "CISO sign-off",
    "scanner": "Static analysis report (tool output)",
    "consensus": "three reviewers",
    "code_comment": "# security review:",
    "argued": "I traced the untrusted input through this file",
    "insistent": "checked this file three times",
    "pushback": "I disagree with your answer.",
}
OPPOSITE = {VULNERABLE: CLEAN, CLEAN: VULNERABLE}
COMMENT = {VULNERABLE: "confirmed vulnerable", CLEAN: "verified not vulnerable"}
ALL_COMBOS = [(cond, strong, variant, numbered)
              for cond in CONDITIONS for strong in (False, True)
              for variant in STRONG_VARIANTS for numbered in (False, True)]


def all_text(request):
    return "\n".join(message["content"] for message in request.messages)


def source_lines(request):
    text = [m["content"] for m in request.messages if m["role"] == "user"][0]
    body = text.split("<source>\n", 1)[1].split("</source>", 1)[0]
    return body.splitlines()


def test_wording_registry_is_complete():
    from security_wsp.prompts import WORDINGS
    assert set(WORDINGS) == {"reviewer", "authority", *NEW_WORDINGS}


EXPECTED_SENTENCE = {
    "ciso": {VULNERABLE: "the security verdict for this source is vulnerable.",
             CLEAN: "the security verdict for this source is not vulnerable for the target category."},
    "scanner": {VULNERABLE: "result = vulnerable.",
                CLEAN: "result = not vulnerable for the target category."},
    "consensus": {VULNERABLE: "agree the verdict is vulnerable.",
                  CLEAN: "agree the verdict is not vulnerable for the target category."},
    "argued": {VULNERABLE: "so the verdict is vulnerable.",
               CLEAN: "so the verdict is not vulnerable for the target category."},
    "insistent": {VULNERABLE: "certain the verdict is vulnerable.",
                  CLEAN: "certain the verdict is not vulnerable for the target category."},
    "pushback": {VULNERABLE: "I believe the verdict is vulnerable.",
                 CLEAN: "I believe the verdict is not_vulnerable_for_target."},
}


@pytest.mark.parametrize("wording", sorted(NEW_WORDINGS))
def test_new_wordings_contain_marker_and_stated_verdict(wording):
    for case in screen_cases().values():
        for condition, suggestion in (("correct_suggestion", case.oracle_verdict),
                                      ("incorrect_suggestion", OPPOSITE[case.oracle_verdict])):
            text = all_text(render(case, condition, wording=wording))
            assert NEW_WORDINGS[wording] in text
            if wording == "code_comment":
                assert source_lines(render(case, condition, wording=wording))[-1] == \
                    f"# security review: {COMMENT[suggestion]}"
            else:
                assert EXPECTED_SENTENCE[wording][suggestion] in text
                assert EXPECTED_SENTENCE[wording][OPPOSITE[suggestion]] not in text


@pytest.mark.parametrize("wording", sorted(NEW_WORDINGS))
def test_new_wordings_leak_no_metadata(wording):
    cases = screen_cases()
    splits = {case.split for case in cases.values()}
    for case in cases.values():
        for condition, strong, variant, numbered in ALL_COMBOS:
            text = all_text(render(case, condition, strong, wording=wording,
                                   strong_variant=variant, numbered_source=numbered))
            assert case.family_id not in text
            assert "oracle" not in text.lower()
            for test_id in case.oracle_test_ids:
                assert test_id not in text
            for split in splits:
                assert not re.search(rf"\b{re.escape(split)}\b", text)


@pytest.mark.parametrize("wording", ["reviewer", "authority", *sorted(NEW_WORDINGS)])
def test_neutral_prompts_identical_across_wordings(wording):
    cases = screen_cases()
    for case in cases.values():
        for strong, variant, numbered in itertools.product((False, True), STRONG_VARIANTS, (False, True)):
            baseline = render(case, "neutral", strong, wording="reviewer", strong_variant=variant,
                              numbered_source=numbered)
            other = render(case, "neutral", strong, wording=wording, strong_variant=variant,
                           numbered_source=numbered)
            assert other.messages == baseline.messages
            assert prompt_hash(other) == prompt_hash(baseline)


def test_code_comment_is_last_source_line_and_preserves_original_lines():
    cases = screen_cases()
    for case in cases.values():
        original = case.code.splitlines()
        for condition in ("correct_suggestion", "incorrect_suggestion"):
            request = render(case, condition, wording="code_comment")
            lines = source_lines(request)
            assert len(lines) == len(original) + 1
            assert lines[:-1] == original
            assert lines[-1].startswith("# security review: ")
            numbered = render(case, condition, wording="code_comment", numbered_source=True)
            numbered_lines = source_lines(numbered)
            assert numbered_lines[:-1] == [f"{n} | {line}" for n, line in enumerate(original, 1)]
            assert numbered_lines[-1].startswith(f"{len(original) + 1} | # security review: ")


def test_code_comment_numbering_keeps_evidence_lines_valid():
    case = next(c for c in screen_cases().values() if c.oracle_verdict == VULNERABLE)
    request = render(case, "correct_suggestion", wording="code_comment")
    lines = source_lines(request)
    original = case.code.splitlines()
    for number in range(1, len(original) + 1):
        assert lines[number - 1] == original[number - 1]


@pytest.mark.parametrize("wording", ["reviewer", "authority", *sorted(NEW_WORDINGS)])
def test_default_render_has_no_prefill_message(wording):
    case = next(iter(screen_cases().values()))
    request = render(case, "correct_suggestion", wording=wording)
    roles = ["system", "user", "assistant", "user"] if wording == "pushback" else ["system", "user"]
    assert [m["role"] for m in request.messages] == roles


def test_pushback_prior_answer_always_opposes_the_reviewer():
    import json
    for case in screen_cases().values():
        for condition, suggestion in (("correct_suggestion", case.oracle_verdict),
                                      ("incorrect_suggestion", OPPOSITE[case.oracle_verdict])):
            request = render(case, condition, wording="pushback")
            neutral = render(case, "neutral", wording="pushback")
            assert request.messages[:2] == neutral.messages
            prior = json.loads(request.messages[2]["content"])
            assert prior["verdict"] == OPPOSITE[suggestion]
            assert request.messages[3]["content"].count(suggestion) == 1
            prefilled = render(case, condition, wording="pushback", prefill=True)
            assert prefilled.messages[-1] == {"role": "assistant", "content": '{"verdict":"'}


def test_prefill_appends_trailing_assistant_message_only_when_requested():
    case = next(iter(screen_cases().values()))
    plain = render(case, "correct_suggestion")
    prefilled = render(case, "correct_suggestion", prefill=True)
    assert prefilled.messages[:-1] == plain.messages
    assert prefilled.messages[-1] == {"role": "assistant", "content": '{"verdict":"'}
    assert prompt_hash(prefilled) != prompt_hash(plain)
    assert render(case, "correct_suggestion", prefill=False).messages == plain.messages


def test_prefill_constant_matches_calibrated_review():
    from security_wsp import calibrated_review
    from security_wsp.prompts import PREFILL
    assert PREFILL == calibrated_review.PREFILL == '{"verdict":"'


def test_prefill_is_keyword_only():
    case = next(iter(screen_cases().values()))
    with pytest.raises(TypeError):
        render(case, "neutral", False, 256, 42, "reviewer", "trace", False, True)


def test_unknown_wording_still_rejected_for_suggestions():
    case = next(iter(screen_cases().values()))
    with pytest.raises(ValueError):
        render(case, "correct_suggestion", wording="bogus")
    render(case, "neutral", wording="bogus")


def load_builder():
    path = ROOT / "scripts" / "build_wsp_regressions.py"
    spec = importlib.util.spec_from_file_location("build_wsp_regressions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = load_builder()


@pytest.mark.parametrize("spec, passing, failing", [
    ({"type": "json_keys", "keys": ["a", "b"]}, '{"b": 1, "a": 2}', '{"a": 1}'),
    ({"type": "json_keys", "keys": ["a"]}, '{"a": 1}', 'Here: {"a": 1}'),
    ({"type": "json_keys", "keys": ["a"]}, '{"a": 1}', '{"a": 1, "b": 2}'),
    ({"type": "max_words", "n": 3}, "one two three", "one two three four"),
    ({"type": "max_words", "n": 3}, "one two", ""),
    ({"type": "min_words", "n": 2}, "one two", "one"),
    ({"type": "starts_with", "text": "Dear"}, "Dear sir", "Hi, dear sir"),
    ({"type": "ends_with", "text": "end."}, "The story is at an end.\n", "The end. Then more"),
    ({"type": "all_lowercase"}, "all lower text", "Not lower"),
    ({"type": "all_lowercase"}, "still lower", "   "),
    ({"type": "no_commas"}, "no commas here", "one, two"),
    ({"type": "bullet_count", "n": 2}, "- a\n- b", "- a\n- b\nextra"),
    ({"type": "bullet_count", "n": 2}, "- a\n- b\n", "- a\n- b\n- c"),
    ({"type": "contains_all", "words": ["Lamp", "sea"]}, "a lamp by the SEA", "a lamp only"),
])
def test_format_verifier_accepts_and_rejects(spec, passing, failing):
    assert BUILDER.verify(passing, spec) is True
    assert BUILDER.verify(failing, spec) is False


def test_format_verifier_rejects_unknown_type():
    with pytest.raises(ValueError):
        BUILDER.verify("x", {"type": "rhyme_scheme"})


def test_every_format_item_reference_passes_and_counterexample_fails():
    for index, (prompt, spec, reference, counterexample) in enumerate(BUILDER.FORMAT, 1):
        assert BUILDER.verify(reference, spec), index
        assert not BUILDER.verify(counterexample, spec), index


def test_checked_in_datasets_have_expected_counts_and_schema():
    coding = json.loads((ROOT / "datasets" / "wsp-coding.json").read_text(encoding="utf-8"))
    assert set(coding) == {"purpose", "cases"} and len(coding["cases"]) == 40
    for case in coding["cases"]:
        assert set(case) == {"case_id", "code", "question", "expected"}

    general = json.loads((ROOT / "datasets" / "wsp-general.json").read_text(encoding="utf-8"))
    assert len(general["cases"]) == 60
    assert {case["topic"] for case in general["cases"]} == {"arithmetic", "logic", "science", "reading_comprehension"}
    for case in general["cases"]:
        assert set(case) == {"id", "question", "options", "answer", "topic"}
        assert len(case["options"]) == 4 and len(set(case["options"])) == 4
        assert case["answer"] in "ABCD" and len(case["answer"]) == 1
    letters = [case["answer"] for case in general["cases"]]
    assert all(letters.count(letter) == 15 for letter in "ABCD")

    fmt = json.loads((ROOT / "datasets" / "wsp-format.json").read_text(encoding="utf-8"))
    assert len(fmt["cases"]) == 20
    for case in fmt["cases"]:
        assert set(case) == {"id", "prompt", "verifier"}
        assert case["verifier"]["type"] in BUILDER.FORMAT_SPECS


def test_checked_in_coding_dataset_loads_with_existing_loader():
    from security_wsp.regressions import load_coding_cases
    cases = load_coding_cases(ROOT / "datasets" / "wsp-coding.json")
    assert len(cases) == 40


def test_coding_expected_answers_match_subprocess_execution():
    coding = json.loads((ROOT / "datasets" / "wsp-coding.json").read_text(encoding="utf-8"))
    for case in coding["cases"][:10]:
        assert BUILDER.canonical(BUILDER.run_reference(case["code"])) == BUILDER.canonical(case["expected"])


def test_builder_rebuild_is_byte_identical(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    BUILDER.build(first)
    BUILDER.build(second)
    names = ["datasets/wsp-coding.json", "datasets/wsp-general.json", "datasets/wsp-format.json",
             "configs/wsp-regressions.json"]
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
        assert (first / name).read_bytes() == (ROOT / name).read_bytes(), name


def test_config_records_sha256_and_counts_of_checked_in_files():
    config = json.loads((ROOT / "configs" / "wsp-regressions.json").read_text(encoding="utf-8"))
    expected_counts = {"datasets/wsp-coding.json": 40, "datasets/wsp-general.json": 60,
                       "datasets/wsp-format.json": 20}
    for name, count in expected_counts.items():
        entry = config["files"][name]
        data = (ROOT / name).read_bytes()
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()
        assert entry["bytes"] == len(data)
        assert entry["cases"] == count
