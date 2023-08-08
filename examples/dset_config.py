from __future__ import annotations

import rich
import vod_configs

if __name__ == "__main__":
    defaults = vod_configs.SearchFactoryDefaults(
        faiss=vod_configs.FaissFactoryConfig(factory="IVF32,Flat"),  # type: ignore
    )

    cfg = {
        "train": {
            "train_queries": [
                {
                    "name": "frank_a.en:train",
                    "link": "frank_a_sections.en:all",
                },
                {
                    "name": "msmarco",
                    "subset": "en",
                    "split": "all",
                    "link": "my-wiki.multi",
                },
            ],
            "val_queries": [
                "frank_a.en:val",
            ],
            "sections": [
                {
                    "name": "frank_a_sections.en:all",
                    "search": {
                        "dense": {
                            "backend": "qdrant",
                        },
                    },
                },
                {
                    "name": "my-wiki.multi",
                    "parts": [
                        "wiki.en",
                        "wiki.fr",
                    ],
                    "search": {
                        "dense": {
                            "backend": "faiss",
                            "factory": "PQ32_512,IVFauto,PQ32",
                        },
                    },
                },
            ],
        },
        "benchmark": [
            {
                "queries": "frank_b.en:val",
                "sections": "frank_b_en_sections:all",
            }
        ],
        "base_search": {
            "engines": {
                "dense": {
                    "backend": "faiss",
                    "factory": "Flat",
                },
                "sparse": {
                    "backend": "elasticsearch",
                },
            },
        },
        "factory": {},
    }
    model = vod_configs.DatasetsConfig(**cfg)
    rich.print(model)

    rich.print("======= RESOLVED CONFIG ========")
    for section in model.train.sections:
        resolved_config = model.resolve_search_config(defaults, section.search)
        rich.print(resolved_config)
