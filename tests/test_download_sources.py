from __future__ import annotations

import unittest

from mikazuki.download_sources import (
    DownloadSources,
    apply_github_prefix,
    install_process_env,
    parse_download_sources,
    pytorch_extra_index_url,
)


class DownloadSourcesTests(unittest.TestCase):
    def test_parse_flat_payload(self):
        sources = parse_download_sources(
            {
                "dry_run": False,
                "pip_index_url": "https://pypi.tuna.tsinghua.edu.cn/simple",
                "pytorch_index_url": "https://mirrors.aliyun.com/pytorch-wheels",
                "hf_endpoint": "https://hf-mirror.com",
                "github_url_prefix": "https://ghfast.top/",
            }
        )
        self.assertIsNotNone(sources)
        assert sources is not None
        self.assertEqual(sources.pip_index_url, "https://pypi.tuna.tsinghua.edu.cn/simple")
        self.assertEqual(sources.github_url_prefix, "https://ghfast.top/")

    def test_parse_nested_payload(self):
        sources = parse_download_sources(
            {"download_sources": {"hf_endpoint": "https://huggingface.co", "pip_index_url": ""}}
        )
        self.assertEqual(sources, DownloadSources(hf_endpoint="https://huggingface.co"))

    def test_parse_empty_returns_none(self):
        self.assertIsNone(parse_download_sources({"dry_run": False}))
        self.assertIsNone(parse_download_sources({"pip_index_url": "  "}))

    def test_apply_github_prefix(self):
        url = "https://github.com/kohya-ss/musubi-tuner.git"
        self.assertEqual(apply_github_prefix(url, None), url)
        self.assertEqual(
            apply_github_prefix(url, "https://ghfast.top"),
            "https://ghfast.top/https://github.com/kohya-ss/musubi-tuner.git",
        )
        prefixed = "https://ghfast.top/" + url
        self.assertEqual(apply_github_prefix(url, "https://ghfast.top/"), prefixed)
        self.assertEqual(apply_github_prefix(prefixed, "https://ghfast.top/"), prefixed)

    def test_pytorch_extra_index_appends_cuda_tag(self):
        self.assertEqual(
            pytorch_extra_index_url("https://mirrors.aliyun.com/pytorch-wheels", "cu128", "fallback"),
            "https://mirrors.aliyun.com/pytorch-wheels/cu128",
        )
        self.assertEqual(
            pytorch_extra_index_url("https://download.pytorch.org/whl/cu130", "cu128", "fallback"),
            "https://download.pytorch.org/whl/cu130",
        )
        self.assertEqual(
            pytorch_extra_index_url(None, "cu128", "https://download.pytorch.org/whl/cu128"),
            "https://download.pytorch.org/whl/cu128",
        )

    def test_parse_proxy_url(self):
        sources = parse_download_sources({"proxy_url": "http://127.0.0.1:7890"})
        self.assertEqual(sources, DownloadSources(proxy_url="http://127.0.0.1:7890"))

    def test_install_process_env_with_proxy_covers_both_cases(self):
        env = install_process_env(DownloadSources(proxy_url="http://127.0.0.1:7890"))
        self.assertIsNotNone(env)
        assert env is not None
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            self.assertEqual(env[key], "http://127.0.0.1:7890")
        self.assertNotIn("HF_ENDPOINT", env)

    def test_install_process_env_without_source_overrides_uses_network_policy(self):
        for sources in (None, DownloadSources()):
            env = install_process_env(sources)
            self.assertNotIn("HF_ENDPOINT", env)
            self.assertIn("HTTP_PROXY", env)
            self.assertIn("HTTPS_PROXY", env)


if __name__ == "__main__":
    unittest.main()
